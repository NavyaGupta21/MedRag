"""Grounded answer generation over retrieved evidence.

The prompt asks the model to cite; the code checks that it did. Those are
separate jobs on purpose. A 1.5B model will not reliably police its own
citations, so nothing here depends on it doing so - the validator downgrades
`grounded` and surfaces the offending sentences instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np

from .config import (
    ARTICLE_ENCODER,
    ATTRIBUTION_FLOOR,
    GENERATE_MAX_NEW_TOKENS,
    GENERATOR_MODEL,
    SCORE_FLOOR,
    TOP_K,
)
from .corpus import ChunkRecord
from .embed import Index, _encode, device
from .retrieve import Hit, format_evidence, search
from .safety import ABSTENTION_MESSAGE, Refusal, screen, with_disclaimer

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = (
    "You are a careful medical reference assistant. You answer strictly from "
    "the numbered evidence snippets you are given.\n"
    "Rules:\n"
    "1. Use only the snippets. Never add facts from your own knowledge.\n"
    "2. End every sentence that makes a clinical claim with the marker of the "
    "snippet it came from, written exactly as [S1], [S2], and so on. The only "
    "valid markers are the [S#] labels shown in the evidence.\n"
    f"3. If the snippets do not support an answer, reply with exactly: {INSUFFICIENT}\n"
    "4. Be concise and factual. Write in plain prose. Do not repeat the snippet "
    "headings, do not use square-bracket placeholders, and do not give "
    "personalized medical advice."
)

# This prompt contains no worked example, and that is deliberate.
#
# A version with a concrete example ("Airway inflammation is the primary driver
# [S1]") caused the 1.5B generator to copy that sentence verbatim into an answer
# about schizotypal personality disorder. Replacing it with an abstract
# placeholder did not help - the model then emitted the placeholder itself,
# opening an answer with "[Statement drawn from snippet 1] [S1]:".
#
# At this model size an example in the prompt becomes content in the output. The
# citation format is described, never demonstrated, and grounding is established
# afterwards by attribute_sentences() rather than trusted to the model.

_GENERATOR: dict[str, tuple] = {}

# Sentences that are framing rather than clinical assertions. These are exempt
# from the citation requirement, so hedging does not count as ungrounded.
_NON_CLAIM_PATTERNS = (
    r"^(the )?(evidence|snippets?|corpus|passages?|context)\b",
    r"^(based on|according to|in summary|overall|however|note that)\b",
    r"^(this|these) (answer|response|summary)\b",
    INSUFFICIENT,
)


@dataclass
class Attribution:
    """A pipeline-computed link from an answer sentence to its source snippet.

    Distinct from a citation: a citation is what the model asserted, an
    attribution is what the evidence supports. They are kept separate so a
    reader always knows which is which.
    """

    sentence: str
    marker: str
    record: ChunkRecord
    score: float


@dataclass
class RAGAnswer:
    """The full result of one question, including why it came out this way."""

    question: str
    answer: str
    retrieved: list[Hit] = field(default_factory=list)
    citations: dict[str, ChunkRecord] = field(default_factory=dict)
    attributions: list[Attribution] = field(default_factory=list)
    abstained: bool = False
    refused: bool = False
    grounded: bool = True
    unsupported_sentences: list[str] = field(default_factory=list)
    dropped_markers: list[str] = field(default_factory=list)
    refusal: Refusal | None = None

    @property
    def answered(self) -> bool:
        return not (self.abstained or self.refused)


def _load_generator():
    """Load the generator once and keep it.

    float16 on an accelerator, float32 on CPU: half precision halves both the
    memory footprint (about 3GB rather than 6GB) and load time, but several
    CPU kernels either lack fp16 paths or fall back to something slower, so
    CPU keeps full precision.
    """
    if "model" not in _GENERATOR:
        target = device()
        dtype = torch.float16 if target.type in ("mps", "cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        model = (
            AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL, torch_dtype=dtype)
            .to(target)
            .eval()
        )
        _GENERATOR["model"] = (tokenizer, model)
    return _GENERATOR["model"]


def build_prompt(question: str, hits: list[Hit]) -> list[dict]:
    """Assemble the chat messages: system rules, numbered evidence, question."""
    evidence = format_evidence(hits)
    user = (
        f"Evidence snippets:\n\n{evidence}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the snippets above, citing each claim with its [S#] marker."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _generate(messages: list[dict], max_new_tokens: int = GENERATE_MAX_NEW_TOKENS) -> str:
    """Greedy decode. Reproducibility matters more than fluency here, and
    sampling is where small models invent drug names and dosages."""
    tokenizer, model = _load_generator()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(device())
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _is_claim(sentence: str) -> bool:
    """Whether a sentence asserts something clinical and so needs a citation.

    A deliberately conservative heuristic: very short fragments and framing
    sentences are exempt. It is meant to catch obvious ungrounded assertions,
    not to adjudicate every sentence.
    """
    stripped = sentence.strip().lower()
    if len(stripped.split()) < 5:
        return False
    return not any(re.search(p.lower(), stripped) for p in _NON_CLAIM_PATTERNS)


def validate_citations(answer: str, hits: list[Hit]) -> tuple[str, dict, list[str], list[str]]:
    """Check the model's citations against the evidence it was actually given.

    Returns (cleaned_answer, citations, unsupported_sentences, dropped_markers).
    Markers pointing outside the evidence set are removed - a small model does
    emit [S9] when it was given eight snippets - and claim-bearing sentences
    with no marker are recorded.
    """
    valid = {f"S{hit.rank + 1}": hit.record for hit in hits}

    dropped = []
    for marker in re.findall(r"\[S(\d+)\]", answer):
        if f"S{marker}" not in valid:
            dropped.append(f"[S{marker}]")
    cleaned = answer
    for marker in set(dropped):
        cleaned = cleaned.replace(marker, "")
    cleaned = re.sub(r" +", " ", cleaned).strip()

    citations = {
        f"[{key}]": record
        for key, record in valid.items()
        if f"[{key}]" in cleaned
    }

    unsupported = [
        sentence
        for sentence in _split_sentences(cleaned)
        if _is_claim(sentence) and not re.search(r"\[S\d+\]", sentence)
    ]
    return cleaned, citations, unsupported, dropped


def attribute_sentences(
    answer: str,
    hits: list[Hit],
    index: Index | None = None,
    floor: float = ATTRIBUTION_FLOOR,
    encode_fn=None,
) -> tuple[list[Attribution], list[str]]:
    """Determine which snippet each uncited claim actually came from.

    Qwen2.5-1.5B does not reliably emit [S#] markers however the prompt is
    written, and pushing harder makes it worse - a concrete worked example gets
    copied into the answer as fact. So rather than trusting the model to cite,
    the pipeline computes the attribution itself and labels it as such.

    Each uncited claim sentence is embedded with the article encoder and
    compared against the retrieved snippets in the index's centered space. The
    best match is returned when it clears `floor`; sentences that match nothing
    are reported as genuinely unsupported.

    Returns (attributions, unsupported_sentences).
    """
    claims = [s for s in _split_sentences(answer) if _is_claim(s)]
    uncited = [s for s in claims if not re.search(r"\[S\d+\]", s)]
    if not uncited or not hits:
        return [], uncited

    encode_fn = encode_fn or (lambda texts: _encode(texts, ARTICLE_ENCODER))
    sentence_vectors = encode_fn(uncited)
    snippet_vectors = encode_fn([hit.record.text for hit in hits])

    # Compare in the same centered space retrieval uses, for the same reason:
    # raw cosines here are dominated by a shared component that carries no
    # information but destroys any absolute threshold.
    # Prefer the index's own center so attribution shares retrieval's space.
    # Fall back to the snippet mean when it is absent or its dimension does not
    # match - a mismatch means the index was built with a different encoder,
    # and silently broadcasting one into the other would produce meaningless
    # scores rather than an error.
    center = snippet_vectors.mean(axis=0)
    if (
        index is not None
        and index.center is not None
        and index.center.shape[0] == snippet_vectors.shape[1]
    ):
        center = index.center

    def centered(matrix):
        shifted = matrix - center
        norms = np.linalg.norm(shifted, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return shifted / norms

    scores = centered(sentence_vectors) @ centered(snippet_vectors).T

    attributions, unsupported = [], []
    for row, sentence in enumerate(uncited):
        best = int(np.argmax(scores[row]))
        score = float(scores[row][best])
        if score >= floor:
            attributions.append(
                Attribution(
                    sentence=sentence,
                    marker=f"[S{hits[best].rank + 1}]",
                    record=hits[best].record,
                    score=score,
                )
            )
        else:
            unsupported.append(sentence)
    return attributions, unsupported


def answer_question(
    question: str,
    index: Index,
    k: int = TOP_K,
    floor: float = SCORE_FLOOR,
    generate_fn=None,
    query_vector=None,
    encode_fn=None,
) -> RAGAnswer:
    """Run the full pipeline, applying all three enforcement layers in order.

    `generate_fn` is injectable so the orchestration can be tested without
    loading a language model. `query_vector` lets a caller supply a
    pre-computed query embedding, which evaluation uses to avoid re-encoding
    the same question for retrieval metrics and for answering.
    """
    generate_fn = generate_fn or _generate

    # Layer 1: intent gate, before any retrieval happens.
    refusal = screen(question)
    if refusal is not None:
        return RAGAnswer(
            question=question,
            answer=with_disclaimer(refusal.message),
            refused=True,
            refusal=refusal,
        )

    hits = search(question, index, k=k, floor=floor, query_vector=query_vector)

    # Layer 2: retrieval-gated abstention. The model is never invoked, so it
    # cannot be talked into using context that retrieval already rejected.
    if not hits:
        return RAGAnswer(
            question=question,
            answer=with_disclaimer(ABSTENTION_MESSAGE),
            abstained=True,
        )

    raw = generate_fn(build_prompt(question, hits))

    # The model may still decide the evidence is inadequate.
    if INSUFFICIENT in raw:
        return RAGAnswer(
            question=question,
            answer=with_disclaimer(ABSTENTION_MESSAGE),
            retrieved=hits,
            abstained=True,
        )

    # Layer 3: post-generation citation validation, then attribution of
    # whatever the model left uncited.
    cleaned, citations, uncited, dropped = validate_citations(raw, hits)
    attributions, unsupported = attribute_sentences(
        cleaned, hits, index, encode_fn=encode_fn
    )

    return RAGAnswer(
        question=question,
        answer=with_disclaimer(cleaned),
        retrieved=hits,
        citations=citations,
        attributions=attributions,
        # Grounded means every clinical claim traces to a retrieved snippet -
        # whether the model cited it or the pipeline attributed it. A sentence
        # matching no snippet is the case that matters, and it stays visible.
        grounded=not unsupported,
        unsupported_sentences=unsupported,
        dropped_markers=dropped,
    )
