"""Seeded sampling of a small slice from the ~430k-chunk MedRAG corpus.

Two sampling rules operate at different levels and do not conflict:

  * The 50/50 split is **across source types** - equal numbers of textbook and
    StatPearls chunks, rather than the corpus's natural ~30/70 split, so both
    are common enough in retrieval and evaluation to compare.
  * Size-proportional selection operates **within** each source type.

Selecting files by size (an `os.stat` call) rather than by chunk count avoids
scanning 5GB of JSONL just to draw a 2000-chunk sample. File size is a close
proxy for chunk count because chunks are of broadly similar length.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    MANIFEST_PATH,
    SAMPLED_CORPUS_PATH,
    SAMPLE_SIZE,
    SEED,
    STATPEARLS_CHUNK_DIR,
    TEXTBOOK_CHUNK_DIR,
)
from .corpus import ChunkRecord, normalize, save_records, write_manifest

_SOURCE_DIRS = {
    "textbook": TEXTBOOK_CHUNK_DIR,
    "statpearls": STATPEARLS_CHUNK_DIR,
}


def _size_table(source_type: str) -> list[tuple[Path, int]]:
    """File paths and byte sizes for one source type. Stats each file once."""
    directory = _SOURCE_DIRS[source_type]
    if not directory.exists():
        raise FileNotFoundError(
            f"Expected corpus directory {directory}. "
            "The corpus/ tree must contain textbooks/chunk and statpearls/chunk."
        )
    table = [(p, p.stat().st_size) for p in sorted(directory.glob("*.jsonl"))]
    if not table:
        raise FileNotFoundError(f"No .jsonl chunk files found under {directory}.")
    return table


def _allocate(table: list[tuple[Path, int]], n: int, rng: random.Random) -> Counter:
    """Draw n file slots with probability proportional to file size."""
    paths = [p for p, _ in table]
    weights = [size for _, size in table]
    return Counter(rng.choices(paths, weights=weights, k=n))


def _sample_from_file(path: Path, count: int, source_type: str, rng: random.Random):
    """Uniformly sample `count` chunks from one file, without replacement.

    Returns (records, capacity) so the caller can redistribute a shortfall when
    a small StatPearls file was allocated more slots than it has chunks.
    """
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    capacity = len(lines)
    take = min(count, capacity)
    chosen = rng.sample(range(capacity), take)
    records = [normalize(json.loads(lines[i]), source_type) for i in sorted(chosen)]
    return records, capacity


def _sample_source(source_type: str, n: int, rng: random.Random):
    """Sample n chunks from one source type, topping up any shortfall.

    A small StatPearls file can be allocated more slots than it holds. The
    shortfall is redistributed over files that have not been drawn yet, which
    keeps every file's draw independent and makes duplicates impossible.
    """
    table = _size_table(source_type)
    records: list[ChunkRecord] = []
    used: set[Path] = set()
    remaining = list(table)
    wanted = n

    while wanted > 0 and remaining:
        allocation = _allocate(remaining, wanted, rng)
        for path, count in allocation.items():
            file_records, _ = _sample_from_file(path, count, source_type, rng)
            records.extend(file_records)
            used.add(path)
        wanted = n - len(records)
        remaining = [(p, size) for p, size in table if p not in used]

    records.sort(key=lambda r: r.chunk_id)
    if len(records) > n:
        keep = sorted(rng.sample(range(len(records)), n))
        records = [records[i] for i in keep]

    files_used = sorted(p.name for p in used)
    return records, files_used


def sample_corpus(
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
    save: bool = True,
) -> list[ChunkRecord]:
    """Draw a balanced, reproducible slice of the corpus.

    Writes `sampled_corpus.jsonl` and `manifest.json` so that no downstream
    stage ever needs to touch the full 5GB corpus again.
    """
    rng = random.Random(seed)
    per_source = n // 2

    all_records: list[ChunkRecord] = []
    manifest_sources = {}
    for source_type in ("textbook", "statpearls"):
        records, files_used = _sample_source(source_type, per_source, rng)
        all_records.extend(records)
        manifest_sources[source_type] = {
            "requested": per_source,
            "sampled": len(records),
            "files_used": len(files_used),
            "files": files_used,
        }

    if save:
        save_records(all_records, SAMPLED_CORPUS_PATH)
        write_manifest(
            MANIFEST_PATH,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "sample_size_requested": n,
                "sample_size_actual": len(all_records),
                "sources": manifest_sources,
            },
        )
    return all_records
