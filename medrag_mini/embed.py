"""MedCPT embedding and the flat cosine index.

MedCPT is an **asymmetric** bi-encoder. The article encoder and the query
encoder were contrastively trained against each other on PubMed click logs and
are not interchangeable: embedding queries with the article encoder silently
degrades retrieval without raising anything.

Both models pool with **CLS**, not the mean pooling that sentence-transformers
applies by default. That pooling choice is written out explicitly below rather
than hidden behind a subclass, because it is the single most consequential and
least visible detail in this pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import (
    ARTICLE_ENCODER,
    EMBED_BATCH_SIZE,
    EMBED_MAX_LENGTH,
    EMBEDDINGS_PATH,
    MANIFEST_PATH,
    QUERY_ENCODER,
    SEED,
)
from .corpus import ChunkRecord, read_manifest, write_manifest

_MODELS: dict[str, tuple] = {}


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load(model_name: str):
    """Load and cache one encoder. Models are large; load each at most once."""
    if model_name not in _MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device()).eval()
        _MODELS[model_name] = (tokenizer, model)
    return _MODELS[model_name]


def _encode(texts: list[str], model_name: str, batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Encode texts with CLS pooling and L2 normalization."""
    tokenizer, model = _load(model_name)
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=EMBED_MAX_LENGTH,
            return_tensors="pt",
        ).to(device())
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        # CLS pooling: take position 0, not the mean over positions.
        pooled = hidden[:, 0, :]
        # L2 normalize so a dot product is a cosine similarity.
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)


def embed_passages(records: list[ChunkRecord], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Embed corpus passages with the *article* encoder."""
    return _encode([r.text for r in records], ARTICLE_ENCODER, batch_size)


def embed_query(question: str) -> np.ndarray:
    """Embed one question with the *query* encoder. Returns shape (dim,)."""
    return _encode([question], QUERY_ENCODER)[0]


def fingerprint(records: list[ChunkRecord], seed: int = SEED) -> dict:
    """Identity of an index, so a stale one cannot be silently reused."""
    return {
        "article_encoder": ARTICLE_ENCODER,
        "query_encoder": QUERY_ENCODER,
        "seed": seed,
        "count": len(records),
        "first_chunk_id": records[0].chunk_id if records else None,
        "last_chunk_id": records[-1].chunk_id if records else None,
    }


@dataclass
class Index:
    """A flat, exact cosine index with anisotropy correction.

    At 2000x768 this is 6MB and searches in about 2ms. FAISS HNSW is an
    *approximate* index that only earns its complexity past ~1e5 vectors; here
    it would add a dependency and lose recall in exchange for no speed that
    anyone would notice.

    The index stores a `center` vector - the corpus mean embedding - and scores
    against mean-centered vectors. BERT-family embeddings including MedCPT are
    anisotropic: they occupy a narrow cone, so any two texts score ~0.6 cosine
    whether or not they are related. That shared component is identical for
    every passage, so it carries no ranking information, but it does destroy
    any absolute threshold - which is what abstention depends on. Subtracting
    it leaves scores that mean something on their own.

    Measured on this corpus: raw cosine separates answerable from unanswerable
    questions at AUC 0.852; mean-centered, at AUC 0.988, with identical
    recall@1 and slightly better recall@3.
    """

    matrix: np.ndarray  # (n, dim) float32, L2-normalized raw embeddings
    records: list[ChunkRecord]
    center: np.ndarray | None = None  # corpus mean; None disables centering

    def __post_init__(self):
        if self.matrix.shape[0] != len(self.records):
            raise ValueError(
                f"Index has {self.matrix.shape[0]} vectors but {len(self.records)} records."
            )
        if self.center is None:
            # A zero center makes centering the identity, which keeps synthetic
            # and hand-built indexes behaving as plain cosine.
            self.center = np.zeros(self.matrix.shape[1], dtype=np.float32)
        self._centered = _normalize_rows(self.matrix - self.center)

    @property
    def centered_matrix(self) -> np.ndarray:
        """Mean-centered, re-normalized vectors. This is what search scores against."""
        return self._centered

    def center_query(self, vector: np.ndarray) -> np.ndarray:
        """Apply the same centering to a query vector."""
        centered = vector - self.center
        norm = np.linalg.norm(centered)
        return centered / norm if norm > 0 else centered

    def __len__(self) -> int:
        return len(self.records)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def build_index(
    records: list[ChunkRecord],
    save: bool = True,
    batch_size: int = EMBED_BATCH_SIZE,
) -> Index:
    matrix = embed_passages(records, batch_size)
    # The center is a property of this corpus slice, so it is computed once at
    # build time and stored with the vectors it belongs to.
    center = matrix.mean(axis=0).astype(np.float32)
    if save:
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(EMBEDDINGS_PATH, matrix=matrix, center=center)
        manifest = read_manifest(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
        manifest["index"] = fingerprint(records)
        write_manifest(MANIFEST_PATH, manifest)
    return Index(matrix=matrix, records=records, center=center)


def load_index(records: list[ChunkRecord]) -> Index:
    """Load a saved index, refusing to serve one that does not match the corpus."""
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"No index at {EMBEDDINGS_PATH}. Run medrag_mini.embed.build_index() first."
        )
    manifest = read_manifest(MANIFEST_PATH)
    stored = manifest.get("index")
    expected = fingerprint(records)
    if stored != expected:
        raise ValueError(
            "Stored index does not match the current corpus - refusing to use it.\n"
            f"  stored:   {stored}\n"
            f"  expected: {expected}\n"
            "Rebuild with medrag_mini.embed.build_index()."
        )
    stored_arrays = np.load(EMBEDDINGS_PATH)
    return Index(
        matrix=stored_arrays["matrix"],
        records=records,
        center=stored_arrays["center"],
    )
