"""
AutoFabric RAG — Hybrid Retriever
Combines FAISS dense retrieval + BM25 sparse retrieval using
Reciprocal Rank Fusion (RRF, k=60) for robust hybrid search.
"""
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Lazy-loading hybrid retriever.
    Call .load() once at startup, then .retrieve(query, top_k) for searches.
    """

    def __init__(
        self,
        index_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model_name
        self._index: Optional[faiss.Index] = None
        self._chunks: List[Dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self._model: Optional[SentenceTransformer] = None
        self._loaded = False

    # ── Initialization ────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load FAISS index, raw chunks, BM25, and embedding model from disk."""
        faiss_file = self.index_path / "index.faiss"
        chunks_file = self.index_path / "chunks.pkl"

        if not faiss_file.exists() or not chunks_file.exists():
            logger.warning(
                f"FAISS index not found at {self.index_path}. "
                "Run /ingest first to populate the index."
            )
            return False

        logger.info("Loading FAISS index...")
        self._index = faiss.read_index(str(faiss_file))

        logger.info("Loading raw chunks...")
        with open(chunks_file, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info("Building BM25 corpus...")
        tokenized = [c["content"].lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self._model = SentenceTransformer(self.embedding_model_name)

        self._loaded = True
        logger.info(f"Retriever ready — {len(self._chunks)} chunks loaded")
        return True

    def reload(self) -> bool:
        """Reload after new documents are ingested."""
        self._loaded = False
        return self.load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ── Dense Retrieval ───────────────────────────────────────────────────────

    def _dense_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Embed query and search FAISS index."""
        query_vec = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        actual_k = min(top_k, len(self._chunks))
        distances, indices = self._index.search(query_vec, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append({
                **self._chunks[idx],
                "dense_score": float(1 / (1 + dist)),  # convert L2 dist → similarity
                "dense_rank": len(results) + 1,
            })
        return results

    # ── Sparse Retrieval (BM25) ───────────────────────────────────────────────

    def _sparse_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """BM25 retrieval over raw chunks."""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        actual_k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                **self._chunks[idx],
                "sparse_score": float(scores[idx]),
                "sparse_rank": rank + 1,
            })
        return results

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).
        score(d) = Σ 1 / (k + rank(d)) across all result lists.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict] = {}

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = {
                **item,
                "dense_score": item.get("dense_score", 0.0),
                "sparse_score": 0.0,
            }

        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid in chunk_map:
                chunk_map[cid]["sparse_score"] = item.get("sparse_score", 0.0)
            else:
                chunk_map[cid] = {
                    **item,
                    "dense_score": 0.0,
                    "sparse_score": item.get("sparse_score", 0.0),
                }

        fused = []
        for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            entry = chunk_map[cid]
            fused.append({
                "content": entry["content"],
                "source": entry["source"],
                "chunk_id": cid,
                "dense_score": round(entry.get("dense_score", 0.0), 4),
                "sparse_score": round(entry.get("sparse_score", 0.0), 4),
                "rrf_score": round(score, 6),
            })
        return fused

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Full hybrid retrieval pipeline:
        dense + sparse → RRF fusion → top_k results
        """
        if not self._loaded:
            logger.error("Retriever not loaded. Call .load() first.")
            return []

        fetch_k = max(top_k * 3, 20)  # fetch more for better fusion coverage
        dense = self._dense_retrieve(query, fetch_k)
        sparse = self._sparse_retrieve(query, fetch_k)
        fused = self._rrf_fusion(dense, sparse)
        return fused[:top_k]


# ── Module-level singleton ────────────────────────────────────────────────────
_retriever_instance: Optional[HybridRetriever] = None


def get_retriever(
    index_path: str = "data/faiss_index/",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> HybridRetriever:
    """Return the module-level retriever singleton, loading if necessary."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever(index_path, embedding_model)
        _retriever_instance.load()
    return _retriever_instance


def reload_retriever(
    index_path: str = "data/faiss_index/",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> HybridRetriever:
    """Force-reload retriever (called after new document ingestion)."""
    global _retriever_instance
    _retriever_instance = HybridRetriever(index_path, embedding_model)
    _retriever_instance.load()
    return _retriever_instance
