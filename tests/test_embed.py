"""Embedding tests.

Tests that need real model weights are marked `slow` and skipped unless
MEDRAG_MINI_SLOW=1, so the default suite stays fast and offline-safe. The
pooling and fingerprint logic - where the real bugs live - is tested without
touching the network at all.
"""

import os

import numpy as np
import pytest
import torch

from medrag_mini import embed
from medrag_mini.corpus import ChunkRecord
from medrag_mini.embed import Index, fingerprint, load_index

slow = pytest.mark.skipif(
    os.environ.get("MEDRAG_MINI_SLOW") != "1",
    reason="needs model weights; set MEDRAG_MINI_SLOW=1",
)


def make_records(n):
    return [
        ChunkRecord(f"doc_{i}", f"text {i}", "textbook", "Book", None, 2) for i in range(n)
    ]


class TestPooling:
    def test_cls_pooling_takes_position_zero_not_the_mean(self, monkeypatch):
        """The bug this guards against is silent: mean pooling also 'works'."""
        hidden = torch.tensor(
            [[[1.0, 0.0], [0.0, 5.0], [0.0, 5.0]]]  # CLS differs sharply from the mean
        )

        class FakeModel:
            def __call__(self, **kwargs):
                return type("Out", (), {"last_hidden_state": hidden})()

            def to(self, *a):
                return self

            def eval(self):
                return self

        class FakeTokenizer:
            def __call__(self, batch, **kwargs):
                return type("Enc", (), {"to": lambda self, d: {}})()

        monkeypatch.setitem(embed._MODELS, "fake", (FakeTokenizer(), FakeModel()))
        out = embed._encode(["anything"], "fake")

        # CLS row [1, 0] normalized is [1, 0]. The mean would be [0.33, 3.33].
        np.testing.assert_allclose(out[0], np.array([1.0, 0.0]), atol=1e-6)

    def test_output_is_l2_normalized(self, monkeypatch):
        hidden = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])

        class FakeModel:
            def __call__(self, **kwargs):
                return type("Out", (), {"last_hidden_state": hidden})()

        class FakeTokenizer:
            def __call__(self, batch, **kwargs):
                return type("Enc", (), {"to": lambda self, d: {}})()

        monkeypatch.setitem(embed._MODELS, "fake2", (FakeTokenizer(), FakeModel()))
        out = embed._encode(["anything"], "fake2")
        assert np.isclose(np.linalg.norm(out[0]), 1.0)
        np.testing.assert_allclose(out[0], np.array([0.6, 0.8]), atol=1e-6)


class TestIndex:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="vectors but"):
            Index(matrix=np.zeros((3, 768), dtype=np.float32), records=make_records(2))

    def test_fingerprint_tracks_corpus_identity(self):
        assert fingerprint(make_records(5)) != fingerprint(make_records(6))
        assert fingerprint(make_records(5), seed=1) != fingerprint(make_records(5), seed=2)
        assert fingerprint(make_records(5)) == fingerprint(make_records(5))

    def test_fingerprint_names_both_encoders(self):
        fp = fingerprint(make_records(1))
        assert fp["article_encoder"] != fp["query_encoder"]

    def test_stale_index_raises_rather_than_being_served(self, tmp_path, monkeypatch):
        records = make_records(4)
        np.save(tmp_path / "e.npy", np.zeros((4, 768), dtype=np.float32))
        (tmp_path / "m.json").write_text('{"index": {"count": 999}}')
        monkeypatch.setattr(embed, "EMBEDDINGS_PATH", tmp_path / "e.npy")
        monkeypatch.setattr(embed, "MANIFEST_PATH", tmp_path / "m.json")
        with pytest.raises(ValueError, match="refusing to use it"):
            load_index(records)

    def test_missing_index_gives_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed, "EMBEDDINGS_PATH", tmp_path / "absent.npy")
        with pytest.raises(FileNotFoundError, match="build_index"):
            load_index(make_records(2))


class TestRealModels:
    @slow
    def test_query_and_article_encoders_are_not_the_same_model(self):
        """MedCPT's asymmetry is the point. If these match, one model was
        loaded twice and retrieval degrades with every other test still green."""
        text = "What causes atrial fibrillation?"
        q = embed._encode([text], embed.QUERY_ENCODER)[0]
        a = embed._encode([text], embed.ARTICLE_ENCODER)[0]
        assert not np.allclose(q, a, atol=1e-4)

    @slow
    def test_embeddings_are_normalized_and_correctly_shaped(self):
        vec = embed.embed_query("What is sepsis?")
        assert vec.shape == (768,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
