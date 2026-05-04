"""
AutoFabric — Supervisor Agent + LangGraph Orchestrator
Entrypoint node that:
  1. Runs governance checks (PII + injection)
  2. Routes to search → summarizer → validator pipeline
  3. Assembles final response with inline [Source: file, score: X] citations
  4. Manages LangSmith run tracing

Graph wiring:
  START → supervisor_entry → search → summarizer → validator → supervisor_final → END
"""
import logging
import time
from typing import Any, Dict, Literal

from langgraph.graph import END, START, StateGraph

from backend.agents.search_agent import search_agent
from backend.agents.state import AgentState
from backend.agents.summarizer_agent import summarizer_agent
from backend.agents.validator_agent import validator_agent
from backend.config import settings
from backend.governance.injection_guard import check_injection
from backend.governance.pii_detector import detect_and_redact

logger = logging.getLogger(__name__)

INJECTION_BLOCK_MSG = (
    "🚫 **Request Blocked** — Your query has been identified as a potential "
    "prompt injection attempt and cannot be processed. "
    "Please rephrase your question."
)


# ── Node: Supervisor Entry (governance + routing) ─────────────────────────────
def supervisor_entry(state: AgentState) -> Dict:
    """
    First node: governance checks.
    - Detects + redacts PII
    - Detects prompt injection
    Returns updated state fields.
    """
    query = state.get("query", "")
    trace = list(state.get("agent_trace", []))

    logger.info(f"[Supervisor] Processing query: {query[:80]}")
    trace.append(f"Supervisor: received query ({len(query)} chars)")

    # ── PII Detection ──────────────────────────────────────────────────────────
    pii_result = detect_and_redact(query)
    pii_detected = pii_result.get("detected", False)
    redacted_query = pii_result.get("redacted_text", query)
    pii_entities = pii_result.get("entities", [])

    if pii_detected:
        entity_types = list({e["type"] for e in pii_entities})
        trace.append(f"Supervisor: 🔒 PII detected ({entity_types}) — query redacted")
        logger.warning(f"[Supervisor] PII detected: {entity_types}")
    else:
        trace.append("Supervisor: ✅ PII check passed")

    # ── Injection Detection ────────────────────────────────────────────────────
    injection_result = check_injection(
        text=query,
        groq_api_key=settings.groq_api_key,
        model=settings.groq_model_fast,
    )
    injection_detected = injection_result.get("is_injection", False)

    if injection_detected:
        reason = injection_result.get("reason", "unknown")
        trace.append(f"Supervisor: 🛑 Injection detected — {reason[:60]}")
        logger.warning(f"[Supervisor] Injection detected: {reason}")

    return {
        "pii_detected": pii_detected,
        "pii_entities": pii_entities,
        "redacted_query": redacted_query,
        "injection_detected": injection_detected,
        "agent_trace": trace,
    }


# ── Node: Supervisor Final (response assembly) ────────────────────────────────
def supervisor_final(state: AgentState) -> Dict:
    """
    Final node: assembles the response with inline source citations.
    Adds [Source: filename, score: X] after each summarized claim block.
    """
    summary = state.get("summary", "")
    chunks = state.get("retrieved_chunks", [])
    trace = list(state.get("agent_trace", []))
    injection_detected = state.get("injection_detected", False)

    # Short-circuit for injection
    if injection_detected:
        trace.append("Supervisor (final): ⛔ injection block — returning error response")
        return {
            "final_response": INJECTION_BLOCK_MSG,
            "sources": [],
            "agent_trace": trace,
        }

    # Build source citations
    sources = []
    for chunk in chunks:
        score = chunk.get("cross_encoder_score", chunk.get("rrf_score", 0.0))
        sources.append({
            "source": chunk.get("source", "unknown"),
            "score": round(score, 4),
            "content_preview": chunk.get("content", "")[:120] + "...",
        })

    # Append source block to summary
    if sources:
        source_lines = "\n".join(
            f"  [{i+1}] {s['source']} (relevance: {s['score']})"
            for i, s in enumerate(sources)
        )
        final_response = summary + f"\n\n**Sources:**\n{source_lines}"
    else:
        final_response = summary

    trace.append(f"Supervisor (final): ✅ assembled response with {len(sources)} sources")
    logger.info(f"[Supervisor] Final response assembled ({len(final_response)} chars)")

    return {
        "final_response": final_response,
        "sources": sources,
        "agent_trace": trace,
    }


# ── Routing Logic ─────────────────────────────────────────────────────────────
def route_after_entry(state: AgentState) -> Literal["search", "supervisor_final"]:
    """Route to search if safe; directly to final if injection detected."""
    if state.get("injection_detected", False):
        return "supervisor_final"
    return "search"


# ── Graph Builder ─────────────────────────────────────────────────────────────
def build_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Wiring:
        START → supervisor_entry
                     │
            [injection?] ──Yes──► supervisor_final → END
                     │
                    No
                     │
                   search → summarizer → validator → supervisor_final → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor_entry", supervisor_entry)
    graph.add_node("search", search_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("supervisor_final", supervisor_final)

    # Edges
    graph.add_edge(START, "supervisor_entry")
    graph.add_conditional_edges(
        "supervisor_entry",
        route_after_entry,
        {
            "search": "search",
            "supervisor_final": "supervisor_final",
        },
    )
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "validator")
    graph.add_edge("validator", "supervisor_final")
    graph.add_edge("supervisor_final", END)

    return graph.compile()


# ── Compiled graph singleton ───────────────────────────────────────────────────
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ── Public Runner ──────────────────────────────────────────────────────────────
def run_graph(query: str, top_k: int = 5) -> AgentState:
    """
    Execute the full agent pipeline and return final state.

    Args:
        query:  Raw user query
        top_k:  Number of chunks to retrieve

    Returns:
        Completed AgentState dict
    """
    initial_state: AgentState = {
        "query": query,
        "top_k": top_k,
        "retrieved_chunks": [],
        "summary": "",
        "validation_result": {},
        "final_response": "",
        "sources": [],
        "agent_trace": [],
        "run_id": None,
        "pii_detected": False,
        "injection_detected": False,
        "pii_entities": [],
        "redacted_query": query,
        "error": None,
    }

    graph = get_graph()
    start = time.perf_counter()
    result = graph.invoke(initial_state)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"[Graph] Completed in {elapsed_ms:.1f}ms")

    return result
