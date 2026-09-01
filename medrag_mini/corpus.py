"""Normalization of the pre-chunked MedRAG corpus into typed records.

Both corpus sources ship pre-chunked as JSONL with `id`, `title`, `content` and
`contents` fields, so there is no chunking stage here - only normalization and
the recovery of the structured metadata that StatPearls hides in its titles.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import STATPEARLS_TITLE_SEP


# StatPearls passages carry their own inline reference markers - "[7]",
# "[41, 42]" - pointing at that article's bibliography, which this pipeline
# does not index. About 28% of sampled chunks contain them.
#
# They must be stripped, and not only for tidiness: the generator is asked to
# cite with [S1]-style markers, and when the evidence text is full of [7] the
# model copies that style instead. The citation validator then finds no valid
# markers and correctly reports the answer as ungrounded. Removing a reference
# the reader cannot follow anyway also removes the collision.
_SOURCE_REFERENCE_MARKER = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def strip_source_references(text: str) -> str:
    """Remove [7] / [41, 42] style markers pointing at an unindexed bibliography."""
    cleaned = _SOURCE_REFERENCE_MARKER.sub("", text)
    # Tidy the spacing those markers leave behind before punctuation.
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


@dataclass(frozen=True)
class ChunkRecord:
    """One retrievable passage, normalized across both corpus sources."""

    chunk_id: str
    text: str
    source_type: str  # "textbook" | "statpearls"
    document: str  # book name, or StatPearls article title
    section: str | None  # StatPearls section heading; None for textbooks
    token_estimate: int

    @property
    def label(self) -> str:
        """Human-readable provenance, used in prompts and evidence displays."""
        if self.section:
            return f"{self.document} - {self.section}"
        return self.document


def parse_statpearls_title(title: str) -> tuple[str, str | None]:
    """Split a StatPearls title into (document, section).

    Titles look like "Article -- Section" or "Article -- Section -- Subsection".
    Splitting is done only on the exact " -- " separator: article names such as
    "Beta-Blockers" and "Non-Hodgkin Lymphoma" contain bare hyphens that must
    survive intact.
    """
    parts = title.split(STATPEARLS_TITLE_SEP)
    document = parts[0].strip()
    if len(parts) == 1:
        return document, None
    # Rejoin any subsection tail so "Treatment / Management -- Medications"
    # stays one section string rather than being silently truncated.
    section = STATPEARLS_TITLE_SEP.join(p.strip() for p in parts[1:]).strip()
    return document, (section or None)


def estimate_tokens(text: str) -> int:
    """Cheap whitespace-based token estimate.

    Deliberately not a real tokenizer: this is only used for display and for
    rough context budgeting, and loading a tokenizer here would make corpus
    normalization depend on transformers.
    """
    return len(text.split())


def normalize(raw: dict, source_type: str) -> ChunkRecord:
    """Turn one raw corpus JSONL line into a ChunkRecord."""
    title = raw.get("title", "").strip()
    content = raw.get("content", "").strip()

    if source_type == "statpearls":
        document, section = parse_statpearls_title(title)
    else:
        # Textbook chunks carry the book name as the whole title and have no
        # section structure.
        document, section = title, None

    # The corpus's own `contents` field is "title. content", which is what
    # MedCPT should embed - the section heading is real retrieval signal.
    text = raw.get("contents") or f"{title}. {content}"

    return ChunkRecord(
        chunk_id=raw["id"],
        text=strip_source_references(text),
        source_type=source_type,
        document=document,
        section=section,
        token_estimate=estimate_tokens(text),
    )


def save_records(records: list[ChunkRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record)) + "\n")


def load_records(path: Path) -> list[ChunkRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"No sampled corpus at {path}. Run medrag_mini.sample.sample_corpus() first."
        )
    with path.open() as fh:
        return [ChunkRecord(**json.loads(line)) for line in fh if line.strip()]


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def read_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No manifest at {path}. Run the sampling step first.")
    return json.loads(path.read_text())
