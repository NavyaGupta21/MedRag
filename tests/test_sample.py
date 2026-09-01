"""Sampling tests.

These run against the real corpus: the properties that matter (determinism,
balance, provenance) are only meaningful over the real file-size distribution,
and sampling is fast enough - under a second - that fixtures would buy nothing.
"""

import json
from collections import Counter

import pytest

from medrag_mini import config
from medrag_mini.sample import _allocate, _size_table, sample_corpus

pytestmark = pytest.mark.skipif(
    not config.CORPUS_DIR.exists(), reason="corpus/ not present"
)


class TestDeterminism:
    def test_same_seed_gives_identical_sample(self):
        a = sample_corpus(n=200, seed=7, save=False)
        b = sample_corpus(n=200, seed=7, save=False)
        assert [r.chunk_id for r in a] == [r.chunk_id for r in b]

    def test_different_seed_gives_different_sample(self):
        a = sample_corpus(n=200, seed=7, save=False)
        b = sample_corpus(n=200, seed=8, save=False)
        assert [r.chunk_id for r in a] != [r.chunk_id for r in b]


class TestBalanceAndIntegrity:
    def test_exact_fifty_fifty_split(self):
        records = sample_corpus(n=200, seed=1, save=False)
        counts = Counter(r.source_type for r in records)
        assert counts["textbook"] == 100
        assert counts["statpearls"] == 100

    def test_no_duplicate_chunks(self):
        records = sample_corpus(n=400, seed=2, save=False)
        assert len(set(r.chunk_id for r in records)) == len(records)

    def test_every_record_is_populated(self):
        records = sample_corpus(n=200, seed=3, save=False)
        assert all(r.text and r.document and r.chunk_id for r in records)


class TestProvenance:
    def test_sampled_chunks_exist_in_the_source_corpus(self):
        """A sampled chunk must be findable in the file its id points at."""
        records = sample_corpus(n=100, seed=4, save=False)
        statpearls = [r for r in records if r.source_type == "statpearls"][:5]
        for record in statpearls:
            article = record.chunk_id.rsplit("_", 1)[0]
            path = config.STATPEARLS_CHUNK_DIR / f"{article}.jsonl"
            assert path.exists(), f"{path} missing for chunk {record.chunk_id}"
            ids = {json.loads(line)["id"] for line in path.open() if line.strip()}
            assert record.chunk_id in ids


class TestAllocation:
    def test_allocation_is_size_proportional(self):
        """Large files must attract proportionally more slots than small ones."""
        table = _size_table("textbook")
        allocation = _allocate(table, 5000, __import__("random").Random(0))
        by_size = sorted(table, key=lambda ps: ps[1])
        smallest, largest = by_size[0][0], by_size[-1][0]
        assert allocation[largest] > allocation[smallest]

    def test_allocation_total_matches_request(self):
        table = _size_table("textbook")
        allocation = _allocate(table, 500, __import__("random").Random(0))
        assert sum(allocation.values()) == 500


class TestArtifacts:
    def test_manifest_records_seed_and_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "SAMPLED_CORPUS_PATH", tmp_path / "c.jsonl")
        monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "m.json")
        import medrag_mini.sample as sample_mod

        monkeypatch.setattr(sample_mod, "SAMPLED_CORPUS_PATH", tmp_path / "c.jsonl")
        monkeypatch.setattr(sample_mod, "MANIFEST_PATH", tmp_path / "m.json")

        sample_corpus(n=100, seed=5, save=True)
        manifest = json.loads((tmp_path / "m.json").read_text())
        assert manifest["seed"] == 5
        assert manifest["sample_size_actual"] == 100
        assert manifest["sources"]["textbook"]["sampled"] == 50
        assert manifest["sources"]["statpearls"]["files_used"] > 0
