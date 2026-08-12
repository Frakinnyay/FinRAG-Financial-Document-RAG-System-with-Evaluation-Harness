"""
Central configuration for FinRAG.
Reads from .env if present, otherwise falls back to sane defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

INDEX_DIR = Path(os.getenv("INDEX_DIR", str(PROCESSED_DIR)))
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunks_metadata.jsonl"

# Chunking parameters
TEXT_CHUNK_SIZE = 900        # characters, not tokens (simple + robust)
TEXT_CHUNK_OVERLAP = 150
MAX_TABLE_ROWS_PER_CHUNK = 25  # split very long tables, repeating header

# Retrieval
TOP_K = 6

# BGE requires an instruction prefix on QUERIES (not on passages) for
# best retrieval quality. This is a well-documented quirk of BGE models.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
