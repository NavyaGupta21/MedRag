"""Evaluation tests.

Metrics are verified against hand-computed fixtures with known geometry, so a
regression in the scoring code is caught without needing model weights.
"""

import numpy as np
import pytest

from medrag_mini import evaluate as ev
from medrag_mini.corpus import ChunkRecord
from medrag_mini.embed import Index
from medrag_mini.evaluate import (
    GoldItem,
    SECTION_TEMPLATES,
    _string_similarity,
    behavioral_metrics,
    build_gold_set,
    calibrate_floor,
    native_context_metrics,
    retrieval_metrics,
)


def record(i, doc="Asthma", section="Etiology", source="statpearls", text=None):
    return ChunkRecord(
        chunk_id=f"article-{i}_0",
        text=text or f"{doc} -- {section}. Passage about {doc} number {i}.",
        source_type=source,
        document=doc,
        section=section,
        token_estimate=8,
    )


class TestGoldSet:
    def test_questions_follow_section_templates(self):
        gold = build_gold_set([record(i, doc=f"Condition{i}") for i in range(12)])
        assert all(item.question == f"what causes condition{i}?".capitalize().replace("what", "What")
                   or item.question.startswith("What causes") for i, item in enumerate(gold))

    def test_ground_truth_points_at_the_source_chunk(self):
        records = [record(i, doc=f"Condition{i}") for i in range(12)]
        gold = build_gold_set(records)
        assert {item.chunk_id for item in gold} <= {r.chunk_id for r in records}
        assert all(item.reference_contexts == [item.reference] for item in gold)

    def test_one_question_per_article_section_pair(self):
        """Two chunks from the same section would otherwise yield the same
        question with two different 'correct' answers, making recall meaningless."""
        records = [record(i, doc="Asthma", section="Etiology") for i in range(15)]
        records += [record(100 + i, doc=f"Other{i}") for i in range(12)]
        gold = build_gold_set(records)
        assert len({(g.document, g.section) for g in gold}) == len(gold)

    def test_administrative_sections_are_excluded(self):
        assert "Review Questions" not in SECTION_TEMPLATES
        assert "Continuing Education Activity" not in SECTION_TEMPLATES

    def test_textbook_chunks_are_not_used(self):
        records = [record(i, source="textbook", section=None) for i in range(20)]
        with pytest.raises(ValueError, match="too few StatPearls"):
            build_gold_set(records)

    def test_too_small_a_sample_fails_loudly(self):
        with pytest.raises(ValueError, match="raise SAMPLE_SIZE"):
            build_gold_set([record(0), record(1)])


class TestRetrievalMetrics:
    def test_perfect_retrieval_scores_one(self, monkeypatch):
        gold = [GoldItem("q", "c1", "ref", ["ref"], "Doc", "Etiology")]
        hit = type("H", (), {"record": type("R", (), {"chunk_id": "c1"})()})()
        monkeypatch.setattr(ev, "search", lambda *a, **k: [hit])
        metrics = retrieval_metrics(gold, None)
        assert metrics["recall@1"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_gold_at_rank_three_scores_correctly(self, monkeypatch):
        gold = [GoldItem("q", "c3", "ref", ["ref"], "Doc", "Etiology")]
        hits = [
            type("H", (), {"record": type("R", (), {"chunk_id": cid})()})()
            for cid in ["c1", "c2", "c3"]
        ]
        monkeypatch.setattr(ev, "search", lambda *a, **k: hits)
        metrics = retrieval_metrics(gold, None)
        assert metrics["recall@1"] == 0.0
        assert metrics["recall@3"] == 1.0
        assert metrics["mrr"] == pytest.approx(1 / 3)

    def test_missing_gold_scores_zero(self, monkeypatch):
        gold = [GoldItem("q", "missing", "ref", ["ref"], "Doc", "Etiology")]
        monkeypatch.setattr(ev, "search", lambda *a, **k: [])
        metrics = retrieval_metrics(gold, None)
        assert metrics["mrr"] == 0.0
        assert metrics["recall@8"] == 0.0


class TestStringSimilarity:
    def test_identical_text_scores_one(self):
        assert _string_similarity("a b c", "a b c") == 1.0

    def test_disjoint_text_scores_zero(self):
        assert _string_similarity("a b c", "x y z") == 0.0

    def test_partial_overlap(self):
        assert _string_similarity("a b", "b c") == pytest.approx(1 / 3)

    def test_empty_is_safe(self):
        assert _string_similarity("", "a") == 0.0


class TestBehavioralMetrics:
    def test_false_answer_rate_counts_negatives_that_retrieved(self, monkeypatch):
        gold = [GoldItem("q", "c1", "r", ["r"], "D", "Etiology")]
        # Everything retrieves: the negatives all produce false answers.
        monkeypatch.setattr(ev, "search", lambda *a, **k: ["hit"])
        metrics = behavioral_metrics(gold, ["off1", "off2"], None)
        assert metrics["false_answer_rate"] == 1.0
        assert metrics["false_abstention_rate"] == 0.0

    def test_negatives_are_not_pre_filtered_to_zero(self, monkeypatch):
        """Regression: passing only the verified-absent negatives makes this
        metric zero by construction, since the questions that retrieve are
        exactly the false answers being counted."""
        import medrag_mini.evaluate as mod

        gold = [GoldItem("What causes asthma?", "c1", "r", ["r"], "D", "Etiology")]
        # Half the off-domain questions retrieve something.
        monkeypatch.setattr(
            mod, "search", lambda q, *a, **k: ["hit"] if q.startswith("leak") else []
        )
        metrics = behavioral_metrics(gold, ["leak1", "leak2", "clean1", "clean2"], None)
        assert metrics["false_answer_rate"] == 0.5

    def test_false_abstention_rate_counts_positives_that_did_not(self, monkeypatch):
        gold = [GoldItem("q", "c1", "r", ["r"], "D", "Etiology")]
        monkeypatch.setattr(ev, "search", lambda *a, **k: [])
        metrics = behavioral_metrics(gold, ["off1"], None)
        assert metrics["false_abstention_rate"] == 1.0
        assert metrics["false_answer_rate"] == 0.0

    def test_advice_questions_are_all_refused(self, monkeypatch):
        monkeypatch.setattr(ev, "search", lambda *a, **k: [])
        metrics = behavioral_metrics(
            [GoldItem("What causes asthma?", "c", "r", ["r"], "D", "Etiology")], [], None
        )
        assert metrics["refusal_accuracy"] == 1.0
        assert metrics["gold_questions_wrongly_refused"] == 0


class TestNativeContextMetrics:
    def test_relevant_passage_at_rank_one(self, monkeypatch):
        gold = [GoldItem("q", "c1", "the quick brown fox", ["the quick brown fox"], "D", "Etiology")]
        hit = type("H", (), {"record": type("R", (), {"text": "the quick brown fox"})()})()
        monkeypatch.setattr(ev, "search", lambda *a, **k: [hit])
        metrics = native_context_metrics(gold, None)
        assert metrics["context_precision"] == 1.0
        assert metrics["context_recall"] == 1.0

    def test_no_relevant_passage_scores_zero(self, monkeypatch):
        gold = [GoldItem("q", "c1", "alpha beta gamma", ["alpha beta gamma"], "D", "Etiology")]
        hit = type("H", (), {"record": type("R", (), {"text": "completely different words here"})()})()
        monkeypatch.setattr(ev, "search", lambda *a, **k: [hit])
        metrics = native_context_metrics(gold, None)
        assert metrics["context_recall"] == 0.0

    def test_empty_retrieval_scores_zero(self, monkeypatch):
        gold = [GoldItem("q", "c1", "r", ["r"], "D", "Etiology")]
        monkeypatch.setattr(ev, "search", lambda *a, **k: [])
        metrics = native_context_metrics(gold, None)
        assert metrics["context_precision"] == 0.0


class TestCalibration:
    def test_finds_the_separating_threshold(self, monkeypatch):
        """Positives at 0.8, negatives at 0.2: any threshold between them is perfect."""
        gold = [GoldItem(f"pos{i}", "c", "r", ["r"], "D", "Etiology") for i in range(3)]
        negatives = [f"neg{i}" for i in range(3)]
        scores = {f"pos{i}": 0.8 for i in range(3)}
        scores.update({f"neg{i}": 0.2 for i in range(3)})

        index = Index(matrix=np.eye(2, dtype=np.float32), records=[
            ChunkRecord("a", "t", "statpearls", "D", "Etiology", 1),
            ChunkRecord("b", "t", "statpearls", "D", "Etiology", 1),
        ])
        monkeypatch.setattr(ev, "embed_query", lambda q: np.array([scores[q], 0], dtype=np.float32))
        monkeypatch.setattr(index, "center_query", lambda v: v)
        monkeypatch.setattr(type(index), "centered_matrix", property(lambda self: np.eye(2, dtype=np.float32)))

        result = calibrate_floor(gold, negatives, index)
        assert result["accuracy_at_best"] == 1.0
        assert result["auc"] == 1.0
        # float32 round-trip, so compare with tolerance.
        assert result["best_threshold"] == pytest.approx(0.8, abs=1e-6)
