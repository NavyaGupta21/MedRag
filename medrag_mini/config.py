"""Central configuration for the medrag_mini pipeline.

Paths resolve relative to the repository root rather than the current working
directory, so the package behaves identically when imported from the notebook,
from pytest, or from a shell in any subdirectory.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Data ---------------------------------------------------------------
CORPUS_DIR = REPO_ROOT / "corpus"
TEXTBOOK_CHUNK_DIR = CORPUS_DIR / "textbooks" / "chunk"
STATPEARLS_CHUNK_DIR = CORPUS_DIR / "statpearls" / "chunk"

ARTIFACT_DIR = REPO_ROOT / "artifacts"
SAMPLED_CORPUS_PATH = ARTIFACT_DIR / "sampled_corpus.jsonl"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
EMBEDDINGS_PATH = ARTIFACT_DIR / "embeddings.npz"

# --- Sampling -----------------------------------------------------------
# Total chunks drawn from the ~430k-chunk corpus, split evenly across the two
# source types. The same code path handles 6000+ if denser coverage is wanted.
SAMPLE_SIZE = 2000
SEED = 1508

SOURCE_TYPES = ("textbook", "statpearls")

# --- Models -------------------------------------------------------------
# MedCPT is an asymmetric bi-encoder: these two models were contrastively
# trained against each other and are not interchangeable.
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

EMBED_MAX_LENGTH = 512
EMBED_BATCH_SIZE = 32
EMBED_DIM = 768

GENERATE_MAX_NEW_TOKENS = 400

# Minimum centered cosine between an answer sentence and a retrieved snippet
# for the pipeline to attribute that sentence to that snippet.
#
# Measured on 40 sentence/passage pairs: a sentence scores 0.529 +/- 0.194
# against the passage it came from and -0.027 +/- 0.086 against an unrelated
# one. The best-accuracy split is 0.084, but 0.20 sits above the highest
# unsupported score observed (0.183), trading a little recall for the
# guarantee that an attribution shown to a reader is one the evidence
# actually supports.
ATTRIBUTION_FLOOR = 0.20

# --- Retrieval ----------------------------------------------------------
TOP_K = 8
MAX_CHUNKS_PER_DOC = 3

# Calibrated against the observed score distributions of the positive and
# negative evaluation sets (see notebook section 7 and evaluate.calibrate_floor).
#
# This is a threshold on the MEAN-CENTERED cosine, not the raw one. Raw MedCPT
# cosines are anisotropic - every embedding sits in a narrow cone, so unrelated
# text still scores ~0.6 and no absolute raw threshold separates in-corpus from
# out-of-corpus questions (measured AUC 0.852, 60% false-answer rate).
# Subtracting the corpus mean before comparing removes that shared component:
# AUC 0.988, 13% false-answer rate, with retrieval accuracy unchanged.
SCORE_FLOOR = 0.33

# --- Separators ---------------------------------------------------------
# StatPearls encodes hierarchy in the title as "Article -- Section [-- Subsection]".
# Split only on this exact separator; article names contain bare hyphens.
STATPEARLS_TITLE_SEP = " -- "
