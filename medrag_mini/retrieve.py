"""Exact cosine retrieval with the two filters that make abstention possible.

Filter order is a real decision, not an accident:

    score floor -> per-document cap -> truncate to k

Capping after truncation would be useless: a single article that monopolizes
the top-k has already crowded out the alternatives by then.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_CHUNKS_PER_DOC, SCORE_FLOOR, TOP_K
from .corpus import ChunkRecord
from .embed import Index, embed_query


@dataclass
class Hit:
    """One retrieved passage with its provenance and score."""

    record: ChunkRecord
    score: float
    rank: int


def search(
    question: str,
    index: Index,
    k: int = TOP_K,
    floor: float = SCORE_FLOOR,
    max_per_doc: int = MAX_CHUNKS_PER_DOC,
    query_vector: np.ndarray | None = None,
) -> list[Hit]:
    """Retrieve the top passages for a question.

    Returns an empty list when nothing clears the floor. That empty list is the
    abstention signal: the generator is never invoked, so it cannot be talked
    into using context that retrieval already judged too weak.
    """
    if len(index) == 0:
        return []

    vector = embed_query(question) if query_vector is None else query_vector
    # Score in the mean-centered space: both sides are L2-normalized there, so
    # this dot product is a cosine similarity with the corpus-wide shared
    # component removed. That removal is what makes `floor` meaningful.
    scores = index.centered_matrix @ index.center_query(vector)

    # Rank generously before filtering: the floor and the per-document cap can
    # each discard candidates, and we still want k survivors where possible.
    candidate_count = min(len(index), max(k * max(max_per_doc, 1) * 4, k))
    top = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
    top = top[np.argsort(-scores[top])]

    hits: list[Hit] = []
    per_doc: dict[str, int] = {}
    for idx in top:
        score = float(scores[idx])
        if score < floor:
            # Scores are sorted descending, so nothing below can qualify either.
            break
        record = index.records[idx]
        if per_doc.get(record.document, 0) >= max_per_doc:
            continue
        per_doc[record.document] = per_doc.get(record.document, 0) + 1
        hits.append(Hit(record=record, score=score, rank=len(hits)))
        if len(hits) == k:
            break
    return hits


def max_score(question: str, index: Index) -> float:
    """Best similarity in the index, ignoring every filter.

    Used to verify that a question really is unanswerable from this corpus,
    rather than assuming it.
    """
    if len(index) == 0:
        return 0.0
    vector = index.center_query(embed_query(question))
    return float(np.max(index.centered_matrix @ vector))


def format_evidence(hits: list[Hit]) -> str:
    """Render hits as the numbered snippets the generator prompt expects."""
    lines = []
    for hit in hits:
        marker = f"[S{hit.rank + 1}]"
        source = "StatPearls" if hit.record.source_type == "statpearls" else "Textbook"
        lines.append(f"{marker} ({source} - {hit.record.label})\n{hit.record.text}")
    return "\n\n".join(lines)
