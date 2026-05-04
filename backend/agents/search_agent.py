"""
AutoFabric — Search Agent Node
Runs hybrid retrieval (FAISS + BM25 → RRF) then cross-encoder reranking.
Stores top-k re-ranked chunks in LangGraph state.
"""
import logging
from typing import Dict

from backend.agents.state import AgentState
from backend.config import settings
from backend.rag.retriever import get_retriever
from backend.rag.reranker import get_reranker

logger = logging.getLogger(__name__)


def search_agent(state: AgentState) -> Dict:
    """
    LangGraph node: SearchAgent
    Input state fields:  query / redacted_query, top_k
    Output state fields: retrieved_chunks, agent_trace
    """
    query = state.get("redacted_query") or state.get("query", "")
    top_k = state.get("top_k", settings.top_k_default)
    trace = list(state.get("agent_trace", []))

    logger.info(f"[SearchAgent] Retrieving top {top_k} chunks for: {query[:80]}")
    trace.append(f"SearchAgent: retrieving top-{top_k} chunks")

    try:
        retriever = get_retriever(
            index_path=settings.faiss_index_path,
            embedding_model=settings.embedding_model,
        )

        if not retriever.is_loaded:
            logger.warning("[SearchAgent] Retriever not loaded — no index found")
            trace.append("SearchAgent: ⚠️ no FAISS index — returning empty chunks")
            return {
                "retrieved_chunks": [],
                "agent_trace": trace,
            }

        # Fetch more for better reranker coverage
        raw_results = retriever.retrieve(query, top_k=top_k * 3)
        logger.info(f"[SearchAgent] Retrieved {len(raw_results)} raw chunks")

        reranker = get_reranker(model_name=settings.reranker_model)
        reranked = reranker.rerank(query, raw_results, top_k=top_k)
        logger.info(f"[SearchAgent] Reranked to {len(reranked)} chunks")

        trace.append(
            f"SearchAgent: ✅ retrieved {len(raw_results)} chunks, "
            f"reranked → top {len(reranked)} "
            f"(top score: {reranked[0].get('cross_encoder_score', 'N/A') if reranked else 'N/A'})"
        )

        return {
            "retrieved_chunks": reranked,
            "agent_trace": trace,
        }

    except Exception as e:
        logger.error(f"[SearchAgent] Error: {e}", exc_info=True)
        trace.append(f"SearchAgent: ❌ error — {str(e)[:80]}")
        return {
            "retrieved_chunks": [],
            "agent_trace": trace,
            "error": str(e),
        }
