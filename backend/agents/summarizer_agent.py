"""
AutoFabric — Summarizer Agent Node
Calls the MCP summarize tool (LLaMA 3.1 8B via Groq) to generate
a concise summary from retrieved chunks + user query.
"""
import logging
from typing import Dict

from backend.agents.state import AgentState
from backend.mcp.mcp_server import mcp_summarize

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000  # cap to avoid Groq token limits


def summarizer_agent(state: AgentState) -> Dict:
    """
    LangGraph node: SummarizerAgent
    Input state fields:  retrieved_chunks, query / redacted_query
    Output state fields: summary, agent_trace
    """
    chunks = state.get("retrieved_chunks", [])
    query = state.get("redacted_query") or state.get("query", "")
    trace = list(state.get("agent_trace", []))

    logger.info(f"[SummarizerAgent] Summarizing {len(chunks)} chunks")
    trace.append(f"SummarizerAgent: summarizing {len(chunks)} retrieved chunks")

    if not chunks:
        msg = "No relevant information was found in the knowledge base for your query."
        trace.append("SummarizerAgent: ⚠️ no chunks — returning empty context message")
        return {
            "summary": msg,
            "agent_trace": trace,
        }

    # Build context from chunks (content + source citation)
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        context_parts.append(f"[{i}] (Source: {source})\n{content}")

    context = "\n\n".join(context_parts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

    try:
        summary = mcp_summarize(context=context, query=query)
        logger.info(f"[SummarizerAgent] Summary generated ({len(summary)} chars)")
        trace.append(f"SummarizerAgent: ✅ summary generated ({len(summary)} chars)")
        return {
            "summary": summary,
            "agent_trace": trace,
        }
    except Exception as e:
        logger.error(f"[SummarizerAgent] Error: {e}", exc_info=True)
        fallback = "Summary generation failed. Please review the retrieved chunks directly."
        trace.append(f"SummarizerAgent: ❌ error — {str(e)[:80]}")
        return {
            "summary": fallback,
            "agent_trace": trace,
            "error": str(e),
        }
