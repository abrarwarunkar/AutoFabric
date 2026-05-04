"""
AutoFabric RAG — Cross-Encoder Re-Ranker
Scores (query, chunk) pairs with ms-marco-MiniLM-L-6-v2 and re-ranks results
for higher precision than embedding-based similarity alone.
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Lazy-loading cross-encoder reranker.
    Uses HuggingFace CrossEncoder for pointwise scoring.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder: {self.model_name}")
                self._model = CrossEncoder(self.model_name, max_length=512)
                logger.info("Cross-encoder loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                raise

    def rerank(self, query: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Re-rank chunks by cross-encoder score.

        Args:
            query:   The user query string
            chunks:  List of chunk dicts (must have 'content' key)
            top_k:   If set, return only top_k results after reranking

        Returns:
            List of chunks sorted by cross_encoder_score descending,
            each enriched with a 'cross_encoder_score' key.
        """
        if not chunks:
            return []

        self._ensure_loaded()

        pairs = [(query, c["content"]) for c in chunks]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            # Graceful degradation: return original order with score=0
            return [
                {**c, "cross_encoder_score": 0.0}
                for c in chunks
            ]

        enriched = []
        for chunk, score in zip(chunks, scores):
            enriched.append({
                **chunk,
                "cross_encoder_score": round(float(score), 4),
            })

        enriched.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        if top_k is not None:
            enriched = enriched[:top_k]

        logger.info(
            f"Reranked {len(chunks)} chunks → "
            f"top score: {enriched[0]['cross_encoder_score'] if enriched else 'N/A'}"
        )
        return enriched


# ── Module-level singleton ────────────────────────────────────────────────────
_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> CrossEncoderReranker:
    """Return the module-level reranker singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(model_name)
    return _reranker_instance
