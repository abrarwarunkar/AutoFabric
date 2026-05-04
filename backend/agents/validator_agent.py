"""
AutoFabric — Validator Agent Node
Calls the MCP validate tool (LLaMA 3.1 8B via Groq) to check if the
generated summary is grounded in the retrieved context.
Appends [UNVERIFIED] warning if hallucination is detected.
"""
import logging
from typing import Dict

from backend.agents.state import AgentState
from backend.mcp.mcp_server import mcp_validate

logger = logging.getLogger(__name__)

UNVERIFIED_TAG = "\n\n⚠️ [UNVERIFIED] This response could not be fully verified against the source documents. Please review the retrieved chunks for accuracy."

MAX_CONTEXT_CHARS = 4000


def validator_agent(state: AgentState) -> Dict:
    """
    LangGraph node: ValidatorAgent
    Input state fields:  summary, retrieved_chunks
    Output state fields: validation_result, summary (possibly tagged), agent_trace
    """
    summary = state.get("summary", "")
    chunks = state.get("retrieved_chunks", [])
    trace = list(state.get("agent_trace", []))

    logger.info("[ValidatorAgent] Validating summary against source context")
    trace.append("ValidatorAgent: checking hallucination + grounding")

    if not summary or not chunks:
        result = {"is_grounded": False, "confidence": 0.0, "flags": ["missing_context"]}
        tagged_summary = summary + UNVERIFIED_TAG if summary else summary
        trace.append("ValidatorAgent: ⚠️ missing summary or chunks — marking unverified")
        return {
            "validation_result": result,
            "summary": tagged_summary,
            "agent_trace": trace,
        }

    # Build context string from chunks
    context_parts = [c.get("content", "") for c in chunks]
    context = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

    try:
        result = mcp_validate(response=summary, context=context)
        is_grounded = result.get("is_grounded", False)
        confidence = result.get("confidence", 0.0)
        flags = result.get("flags", [])

        log_msg = (
            f"ValidatorAgent: ✅ grounded={is_grounded}, confidence={confidence:.2f}, flags={flags}"
        )
        logger.info(f"[ValidatorAgent] {log_msg}")
        trace.append(log_msg)

        final_summary = summary
        if not is_grounded:
            final_summary = summary + UNVERIFIED_TAG

        return {
            "validation_result": result,
            "summary": final_summary,
            "agent_trace": trace,
        }

    except Exception as e:
        logger.error(f"[ValidatorAgent] Error: {e}", exc_info=True)
        result = {"is_grounded": False, "confidence": 0.0, "flags": ["validation_error"]}
        trace.append(f"ValidatorAgent: ❌ error — {str(e)[:80]}")
        return {
            "validation_result": result,
            "summary": summary + UNVERIFIED_TAG,
            "agent_trace": trace,
            "error": str(e),
        }
