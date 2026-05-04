"""
AutoFabric Agent State Schema
Shared TypedDict state for the LangGraph StateGraph.
"""
from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Shared mutable state passed between all LangGraph nodes."""

    # Input
    query: str
    top_k: int

    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]

    # Generation
    summary: str

    # Validation
    validation_result: Dict[str, Any]

    # Output assembly
    final_response: str
    sources: List[Dict[str, Any]]

    # Tracing
    agent_trace: List[str]
    run_id: Optional[str]

    # Governance
    pii_detected: bool
    injection_detected: bool
    pii_entities: List[Dict]
    redacted_query: str

    # Error handling
    error: Optional[str]
