import json
import sys
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH, METADATA_PATH, EMBED_MODEL, TOP_K, BGE_QUERY_PREFIX,
)


class Retriever:
    def __init__(self):
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"{FAISS_INDEX_PATH} not found. Run ingest/build_index.py first."
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.metadata = []
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        self.model = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, top_k: int = TOP_K, company_filter: str = None):
        """
        Returns (results, retrieval_latency_ms).
        results: list of dicts with text + metadata + similarity score.
        """
        t0 = time.perf_counter()

        query_with_prefix = BGE_QUERY_PREFIX + query
        q_emb = self.model.encode(
            [query_with_prefix], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        # Over-fetch if we're going to filter by company, then trim
        fetch_k = top_k * 5 if company_filter else top_k
        scores, indices = self.index.search(q_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if company_filter and meta["company"].upper() != company_filter.upper():
                continue
            results.append({
                "text": meta["text"],
                "company": meta["company"],
                "fiscal_year": meta["fiscal_year"],
                "source_file": meta["source_file"],
                "page_number": meta["page_number"],
                "chunk_type": meta["chunk_type"],
                "score": float(score),
            })
            if len(results) >= top_k:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms
