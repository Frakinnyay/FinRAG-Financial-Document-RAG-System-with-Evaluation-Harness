"""
Embeds every chunk with BAAI/bge-small-en-v1.5 and builds a FAISS
cosine-similarity index (IndexFlatIP on L2-normalized vectors).

Run after pdf_parser.py has produced data/processed/raw_chunks.jsonl:
    python ingest/build_index.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, FAISS_INDEX_PATH, METADATA_PATH, EMBED_MODEL,
)


def load_chunks(jsonl_path: Path):
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index():
    raw_chunks_path = PROCESSED_DIR / "raw_chunks.jsonl"
    if not raw_chunks_path.exists():
        raise FileNotFoundError(
            f"{raw_chunks_path} not found. Run ingest/pdf_parser.py first."
        )

    chunks = load_chunks(raw_chunks_path)
    print(f"Loaded {len(chunks)} chunks. Loading embedding model {EMBED_MODEL} ...")

    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]

    print("Embedding chunks (this can take a few minutes on CPU) ...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # required for cosine sim via inner product
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nIndex built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved index      -> {FAISS_INDEX_PATH}")
    print(f"Saved metadata   -> {METADATA_PATH}")


if __name__ == "__main__":
    build_index()
