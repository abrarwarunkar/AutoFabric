"""
AutoFabric Observability — LangSmith Tracer + In-Memory Metrics
Wraps LangSmith run lifecycle and accumulates platform-level metrics.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── In-memory metrics (reset on restart) ──────────────────────────────────────
_metrics = {
    "total_queries": 0,
    "total_latency_ms": 0.0,
    "cache_hits": 0,
    "pii_detected_count": 0,
    "injection_detected_count": 0,
}


def record_query(
    latency_ms: float,
    cache_hit: bool = False,
    pii_detected: bool = False,
    injection_detected: bool = False,
):
    """Update in-memory metrics after every query."""
    _metrics["total_queries"] += 1
    _metrics["total_latency_ms"] += latency_ms
    if cache_hit:
        _metrics["cache_hits"] += 1
    if pii_detected:
        _metrics["pii_detected_count"] += 1
    if injection_detected:
        _metrics["injection_detected_count"] += 1


def get_metrics() -> Dict:
    """Return aggregated platform metrics."""
    n = _metrics["total_queries"]
    return {
        "total_queries": n,
        "avg_latency_ms": round(_metrics["total_latency_ms"] / n, 2) if n > 0 else 0.0,
        "cache_hit_rate": round(_metrics["cache_hits"] / n, 4) if n > 0 else 0.0,
        "pii_detection_rate": round(_metrics["pii_detected_count"] / n, 4) if n > 0 else 0.0,
        "injection_detection_rate": round(_metrics["injection_detected_count"] / n, 4) if n > 0 else 0.0,
    }


# ── LangSmith Tracer ───────────────────────────────────────────────────────────
class AutoFabricTracer:
    """
    Thin wrapper around LangSmith Client for run tracing.
    Gracefully no-ops if LANGCHAIN_API_KEY is not configured.
    """

    def __init__(self, api_key: str = "", project: str = "autofabric"):
        self.project = project
        self._client = None
        self._enabled = False

        if api_key:
            try:
                from langsmith import Client
                self._client = Client(api_key=api_key)
                self._enabled = True
                logger.info(f"LangSmith tracing enabled — project: {project}")
            except Exception as e:
                logger.warning(f"LangSmith init failed: {e} — tracing disabled")
        else:
            logger.info("LANGCHAIN_API_KEY not set — LangSmith tracing disabled")

    def start_run(
        self,
        query: str,
        top_k: int = 5,
        model: str = "llama-3.3-70b-versatile",
    ) -> Optional[str]:
        """Start a LangSmith root run. Returns run_id or None."""
        if not self._enabled:
            return None
        try:
            import uuid
            run_id = str(uuid.uuid4())
            self._client.create_run(
                id=run_id,
                name="autofabric_query",
                run_type="chain",
                inputs={"query": query, "timestamp": datetime.now(timezone.utc).isoformat()},
                extra={"metadata": {"model": model, "top_k": top_k, "project": self.project}},
                project_name=self.project,
            )
            return run_id
        except Exception as e:
            logger.warning(f"LangSmith start_run failed: {e}")
            return None

    def log_child(
        self,
        parent_run_id: str,
        node_name: str,
        inputs: Dict,
        outputs: Dict,
        run_type: str = "chain",
    ) -> None:
        """Log a child run (agent node) under the parent query run."""
        if not self._enabled or not parent_run_id:
            return
        try:
            import uuid
            child_id = str(uuid.uuid4())
            self._client.create_run(
                id=child_id,
                name=node_name,
                run_type=run_type,
                inputs=inputs,
                parent_run_id=parent_run_id,
                project_name=self.project,
            )
            self._client.update_run(child_id, outputs=outputs)
        except Exception as e:
            logger.warning(f"LangSmith log_child failed for {node_name}: {e}")

    def end_run(
        self,
        run_id: str,
        final_response: str,
        sources: list,
        validation_result: Dict,
        pii_detected: bool,
        latency_ms: float,
        cache_hit: bool,
    ) -> None:
        """Complete the LangSmith root run with outputs."""
        if not self._enabled or not run_id:
            return
        try:
            self._client.update_run(
                run_id,
                outputs={
                    "final_response": final_response,
                    "sources": sources,
                    "validation_result": validation_result,
                    "pii_detected": pii_detected,
                    "latency_ms": latency_ms,
                    "cache_hit": cache_hit,
                },
            )
        except Exception as e:
            logger.warning(f"LangSmith end_run failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled


# ── Module-level singleton ─────────────────────────────────────────────────────
_tracer_instance: Optional[AutoFabricTracer] = None


def get_tracer(api_key: str = "", project: str = "autofabric") -> AutoFabricTracer:
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = AutoFabricTracer(api_key=api_key, project=project)
    return _tracer_instance
