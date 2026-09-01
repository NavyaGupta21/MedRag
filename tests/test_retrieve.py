"""Retrieval tests over a synthetic index with known geometry.

Hand-built unit vectors make every expected score exact, so these tests assert
behavior rather than approximately reproducing model output.
"""

import numpy as np
import pytest

from medrag_mini.corpus import ChunkRecord
from medrag_mini.embed import Index
from medrag_mini.retrieve import format_evidence, search


def unit(*values):
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def index():
    """Six vectors at known angles from the query direction [1, 0, 0].

    Cosines, in order: 1.00, 0.98, 0.95, 0.89, 0.45, 0.00
    Documents:          A     A     A     A     B     C
    """
    vectors = np.vstack(
        [
            unit(1, 0, 0),
            unit(1, 0.2, 0),
            unit(1, 0.33, 0),
            unit(1, 0.5, 0),
            unit(0.45, 0.893, 0),
            unit(0, 1, 0),
        ]
    )
    docs = ["A", "A", "A", "A", "B", "C"]
    records = [
        ChunkRecord(f"c{i}", f"text {i}", "statpearls", doc, "Etiology", 2)
        for i, doc in enumerate(docs)
    ]
    return Index(matrix=vectors.astype(np.float32), records=records)


QUERY = np.array([1, 0, 0], dtype=np.float32)


class TestScoreFloor:
    def test_sub_threshold_hits_are_dropped(self, index):
        hits = search("q", index, k=6, floor=0.9, max_per_doc=10, query_vector=QUERY)
        assert all(h.score >= 0.9 for h in hits)
        assert len(hits) == 3

    def test_everything_below_floor_returns_empty(self, index):
        """The abstention signal: a query pointing where the corpus has nothing.

        Every indexed vector lies in the x-y plane, so a z-axis query is
        orthogonal to all of them and scores 0.0 across the board.
        """
        off_topic = np.array([0, 0, 1], dtype=np.float32)
        hits = search("q", index, k=6, floor=0.3, max_per_doc=10, query_vector=off_topic)
        assert hits == []

    def test_zero_floor_admits_everything(self, index):
        hits = search("q", index, k=6, floor=0.0, max_per_doc=10, query_vector=QUERY)
        assert len(hits) == 6


class TestPerDocumentCap:
    def test_one_document_cannot_monopolize_results(self, index):
        """Document A holds the 4 best vectors; the cap must let B and C in."""
        hits = search("q", index, k=6, floor=0.0, max_per_doc=3, query_vector=QUERY)
        docs = [h.record.document for h in hits]
        assert docs.count("A") == 3
        assert "B" in docs and "C" in docs

    def test_cap_applies_before_truncation_to_k(self, index):
        """With cap-after-truncate, k=3 would return three A chunks and no
        diversity. Capping first is what makes room for B."""
        hits = search("q", index, k=3, floor=0.0, max_per_doc=2, query_vector=QUERY)
        docs = [h.record.document for h in hits]
        assert docs == ["A", "A", "B"]


class TestOrderingAndShape:
    def test_scores_are_descending(self, index):
        hits = search("q", index, k=6, floor=0.0, max_per_doc=10, query_vector=QUERY)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_contiguous_from_zero(self, index):
        hits = search("q", index, k=6, floor=0.0, max_per_doc=3, query_vector=QUERY)
        assert [h.rank for h in hits] == list(range(len(hits)))

    def test_k_is_respected(self, index):
        hits = search("q", index, k=2, floor=0.0, max_per_doc=10, query_vector=QUERY)
        assert len(hits) == 2

    def test_empty_index_returns_empty(self):
        empty = Index(matrix=np.zeros((0, 3), dtype=np.float32), records=[])
        assert search("q", empty, query_vector=QUERY) == []


class TestFormatEvidence:
    def test_markers_are_one_indexed_and_labelled(self, index):
        hits = search("q", index, k=2, floor=0.0, max_per_doc=10, query_vector=QUERY)
        text = format_evidence(hits)
        assert "[S1]" in text and "[S2]" in text and "[S3]" not in text
        assert "StatPearls" in text and "Etiology" in text
