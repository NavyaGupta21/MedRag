# medrag_mini

A small, fully offline medical RAG pipeline over a random slice of the MedRAG corpus,
presented as an annotated notebook walkthrough backed by a testable Python package.

No API keys. No hosted models. Nothing leaves the machine.

> **This is an educational reference tool, not a clinical device.** It does not give
> medical advice, and its evaluation measures whether retrieval works and whether the
> system abstains honestly — not whether its answers are clinically correct.

## Quick start

```bash
pip install -r requirements-mini.txt
jupyter notebook notebooks/medrag_mini_walkthrough.ipynb
```

Optional, for the RAGAS metrics:

```bash
./scripts/setup_eval_venv.sh
```

**First run downloads ~3.9GB** of model weights: MedCPT's two encoders (~440MB each) and
Qwen2.5-1.5B-Instruct (~3GB). After that everything is cached and offline.

## What it does

```
corpus/*.jsonl  --sample-->  2,000 chunks + manifest
                --embed-->   6MB float32 index + corpus mean
question  --safety gate-->   [refused]
          --retrieve-->      [nothing above floor -> abstained, model never invoked]
          --generate-->      --validate citations-->  RAGAnswer
```

The design centers on three enforcement layers that hold regardless of whether the
language model cooperates:

1. **Intent gate** — rule-based, pre-retrieval. Refuses personal clinical decisions
   ("should I take…", "do I have…") while leaving reference questions about overdose,
   contraindications, and dosing ranges answerable.
2. **Retrieval-gated abstention** — when nothing clears the evidence floor, the generator
   is never called. A model that never sees weak context cannot be talked into using it.
3. **Citation validation and attribution** — `[S#]` markers are parsed and checked against
   the evidence actually supplied; invalid markers are dropped. Because the 1.5B generator
   does not reliably cite at all, the pipeline then computes attribution itself, matching
   each uncited claim to the snippet it came from. Model-asserted *citations* and
   pipeline-computed *attributions* are reported separately, and a claim matching no
   snippet stays visible as unsupported.

## Layout

| Path | Purpose |
|---|---|
| `medrag_mini/config.py` | All constants: paths, sample size, model ids, score floor |
| `medrag_mini/corpus.py` | `ChunkRecord`, StatPearls title parsing, reference stripping |
| `medrag_mini/sample.py` | Seeded, size-proportional, source-balanced sampling |
| `medrag_mini/embed.py` | MedCPT dual encoders, CLS pooling, flat index with centering |
| `medrag_mini/retrieve.py` | Exact cosine search, score floor, per-document cap |
| `medrag_mini/safety.py` | Intent-gate rules, disclaimer, refusal and abstention text |
| `medrag_mini/generate.py` | Prompt assembly, local generation, citation validation |
| `medrag_mini/evaluate.py` | Gold-set construction, retrieval/behavioral/RAGAS metrics |
| `scripts/build_notebook.py` | Regenerates the walkthrough notebook |
| `scripts/ragas_score.py` | RAGAS scoring, run under the isolated venv |
| `notebooks/` | The walkthrough |
| `tests/` | 110 tests; `MEDRAG_MINI_SLOW=1` also runs the weight-loading ones |

```bash
python -m pytest tests/ -q
```

## Four findings worth knowing

**Raw MedCPT cosine scores cannot support abstention.** "What is the capital of France?"
scored 0.665 against this medical corpus while genuine gold matches scored 0.64–0.68 — the
populations overlap almost entirely. This is BERT-family anisotropy: embeddings occupy a
narrow cone, so everything is similar to everything. Subtracting the corpus mean before
scoring moves separation from AUC 0.852 to 0.988 and cuts the false-answer rate on
unanswerable questions from 60% to 13%, with recall@1 unchanged. The index therefore
stores the corpus mean and scores in centered space.

**StatPearls' own reference markers broke citation grounding.** Inline `[7]` and `[41, 42]`
markers appear in 28% of sampled chunks. The generator copied that style instead of the
requested `[S1]` markers, so the validator found nothing valid and every answer read as
ungrounded. They are stripped at normalization.

**A 1.5B model cannot be prompted into citing reliably — and pushing harder is dangerous.**
Qwen2.5-1.5B emitted no `[S#]` markers regardless of prompt phrasing. A variant containing
a concrete worked example ("Airway inflammation is the primary driver [S1]") made the model
copy that sentence verbatim into an answer about schizotypal personality disorder. An
abstract placeholder fared no better — the model emitted the placeholder itself, opening an
answer with "[Statement drawn from snippet 1] [S1]:". At this size an example in the prompt
becomes content in the output, so the prompt now contains none: the format is described,
never demonstrated. Grounding rests on computed attribution with a measured floor (0.20;
supported sentences score 0.529 ± 0.194 against their source passage, unsupported ones
−0.027 ± 0.086).

**RAGAS cannot share this environment.** It pins older `langchain` and `openai` releases,
so installing it would downgrade them system-wide. It runs in a no-system-site-packages
venv instead, communicating over JSON — the pipeline never imports it and works without it.

## Deliberate omissions

No BM25, no reciprocal rank fusion, no cross-encoder reranking, no FAISS. Each is standard
in production medical RAG and each is discussed in the notebook's final section — they were
left out because a 2,000-vector exact index makes them unnecessary complexity here, not
because they lack value at scale.

Answer correctness is not measured. RAGAS faithfulness and answer relevancy need a
competent LLM judge, which is unavailable offline; a 1.5B model grading its own output
would be both unreliable and circular. Embedding similarity is not a substitute — it
rewards paraphrase and cannot tell "aspirin is contraindicated" from "aspirin is
indicated."

## Relationship to the parent repo

This is a standalone pipeline built alongside the original MedRAG code in `src/`, which
implements the full published system (pyserini BM25, MedCPT reranking, RRF fusion,
i-MedRAG follow-up querying, OpenAI generation). `medrag_mini` shares no code with it and
has its own dependency file; the two do not interfere.

Design and implementation notes live in `docs/superpowers/specs/`.
