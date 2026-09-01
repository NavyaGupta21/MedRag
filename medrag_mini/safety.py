"""Deterministic safety rules for a medical reference pipeline.

Everything here is rule-based and readable on one screen. That is the point: at
1.5B parameters, a rule you can audit is more trustworthy than a prediction you
cannot, and these rules must hold regardless of whether the generator complies
with its instructions.

The pipeline enforces three layers, of which this module is the first and third:

  1. an intent gate that fires *before* retrieval,
  2. retrieval-gated abstention (see retrieve.search returning no hits),
  3. constants appended in code, never requested from the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DISCLAIMER = (
    "This is an educational reference generated from a small indexed corpus. "
    "It is not medical advice, not a diagnosis, and not a substitute for a "
    "qualified clinician."
)

ABSTENTION_MESSAGE = (
    "There is insufficient evidence in the indexed corpus to answer this "
    "question. Rather than answer from memory, this pipeline declines. The "
    "corpus is a small random sample of the full medical corpus, so many "
    "legitimate questions fall outside it."
)

REFUSAL_MESSAGE = (
    "This question asks for a personal clinical decision - what someone should "
    "take, do, or stop. This pipeline is a reference tool over textbook and "
    "StatPearls passages and cannot weigh an individual's history, medications, "
    "or circumstances. Please raise it with a qualified clinician or pharmacist. "
    "A general, non-personalized question on the same topic is fine to ask."
)

# Patterns matching requests for a personal clinical decision, as opposed to
# requests for reference knowledge. The distinction being drawn is
# "what should I do" versus "what is known", not topic sensitivity: questions
# about overdose, self-harm or contraindications are legitimate reference
# questions and must remain answerable.
PERSONAL_ADVICE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bshould i\b", "asks what the user personally should do"),
    (r"\bshould my\b", "asks what a specific person should do"),
    (r"\bcan i (take|stop|start|use|mix|double|skip)\b", "asks permission for a personal action"),
    (r"\bis it safe for me\b", "asks about the user's personal safety"),
    (r"\bhow much .* should i\b", "asks for a personal dose"),
    (r"\bwhat dose should\b", "asks for a dosing decision"),
    (r"\bmy (doctor|physician|gp|consultant) (said|told|prescribed)\b", "concerns the user's own care"),
    (r"\bi (have|has|am taking|was prescribed|was diagnosed)\b", "describes the user's own condition"),
    (r"\bmy (symptoms|test results?|blood work|diagnosis|prescription)\b", "concerns the user's own results"),
    (r"\bdo i (have|need)\b", "asks for a personal diagnosis"),
    (r"\bdiagnose me\b", "asks for a personal diagnosis"),
)


@dataclass(frozen=True)
class Refusal:
    """A pre-retrieval refusal, carrying the rule that produced it."""

    message: str
    matched_pattern: str
    reason: str


def screen(question: str) -> Refusal | None:
    """Check a question against the personal-advice rules.

    Returns a Refusal to be surfaced without retrieval or generation, or None
    to let the question proceed.
    """
    lowered = question.lower().strip()
    for pattern, reason in PERSONAL_ADVICE_PATTERNS:
        if re.search(pattern, lowered):
            return Refusal(message=REFUSAL_MESSAGE, matched_pattern=pattern, reason=reason)
    return None


def with_disclaimer(answer: str) -> str:
    """Append the disclaimer in code.

    Never asked of the model, so it cannot be forgotten, paraphrased away, or
    dropped when the model runs out of output tokens.
    """
    return f"{answer.rstrip()}\n\n---\n{DISCLAIMER}"
