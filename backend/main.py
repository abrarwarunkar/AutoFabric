"""
AutoFabric — FastAPI Application Entry Point
Endpoints:
  POST /query   — Full agent pipeline with cache, governance, RAG
  POST /ingest  — Document upload and FAISS index update
  GET  /metrics — Platform observability stats
  GET  /health  — System health check
"""
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agents.supervisor import run_graph
from backend.cache.redis_cache import get_cache
from backend.config import settings
from backend.observability.tracer import get_tracer, get_metrics, record_query
from backend.rag.ingestor import ingest_documents
from backend.rag.retriever import get_retriever, reload_retriever

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AutoFabric starting up...")
    # Pre-load retriever if index exists
    try:
        retriever = get_retriever(settings.faiss_index_path, settings.embedding_model)
        if retriever.is_loaded:
            logger.info(f"FAISS index loaded: {retriever.chunk_count} chunks")
        else:
            logger.warning("No FAISS index found — upload documents via POST /ingest")
    except Exception as e:
        logger.warning(f"Retriever preload skipped: {e}")

    # Init tracer
    get_tracer(api_key=settings.langchain_api_key, project=settings.langchain_project)

    yield
    logger.info("AutoFabric shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AutoFabric API",
    description="Enterprise AI Agent Orchestration Platform — Hybrid RAG + LangGraph + MCP",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    validation: Dict[str, Any]
    pii_detected: bool
    injection_detected: bool
    agent_trace: List[str]
    latency_ms: float
    cache_hit: bool


class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    total_chunks: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    redis: bool
    faiss_loaded: bool
    faiss_chunks: int
    langsmith_enabled: bool


# ── POST /query ───────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse, tags=["Inference"])
async def query_endpoint(request: QueryRequest):
    """
    Full agent pipeline:
    1. Check Redis cache
    2. Run governance (PII + injection)
    3. Run LangGraph: search → summarize → validate → assemble
    4. Cache result and return
    """
    start_time = time.perf_counter()
    cache = get_cache(settings.redis_url, settings.cache_ttl)
    tracer = get_tracer()

    # ── Cache check ────────────────────────────────────────────────────────────
    cached = cache.get(request.query, request.top_k)
    if cached:
        cached["cache_hit"] = True
        latency_ms = (time.perf_counter() - start_time) * 1000
        cached["latency_ms"] = round(latency_ms, 2)
        record_query(
            latency_ms=latency_ms,
            cache_hit=True,
            pii_detected=cached.get("pii_detected", False),
            injection_detected=cached.get("injection_detected", False),
        )
        logger.info(f"Cache hit for query: {request.query[:60]}")
        return QueryResponse(**cached)

    # ── LangSmith run start ────────────────────────────────────────────────────
    run_id = tracer.start_run(
        query=request.query,
        top_k=request.top_k,
        model=settings.groq_model_reasoning,
    )

    # ── Agent graph ────────────────────────────────────────────────────────────
    try:
        final_state = run_graph(query=request.query, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Agent graph error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # ── Assemble response ──────────────────────────────────────────────────────
    response_data = {
        "answer": final_state.get("final_response", "No response generated."),
        "sources": final_state.get("sources", []),
        "validation": final_state.get("validation_result", {}),
        "pii_detected": final_state.get("pii_detected", False),
        "injection_detected": final_state.get("injection_detected", False),
        "agent_trace": final_state.get("agent_trace", []),
        "latency_ms": latency_ms,
        "cache_hit": False,
    }

    # ── Cache result ───────────────────────────────────────────────────────────
    if not response_data["injection_detected"]:
        cache.set(request.query, response_data, request.top_k)

    # ── Record metrics ─────────────────────────────────────────────────────────
    record_query(
        latency_ms=latency_ms,
        cache_hit=False,
        pii_detected=response_data["pii_detected"],
        injection_detected=response_data["injection_detected"],
    )

    # ── LangSmith end run ──────────────────────────────────────────────────────
    tracer.end_run(
        run_id=run_id,
        final_response=response_data["answer"],
        sources=response_data["sources"],
        validation_result=response_data["validation"],
        pii_detected=response_data["pii_detected"],
        latency_ms=latency_ms,
        cache_hit=False,
    )

    logger.info(f"Query completed in {latency_ms}ms")
    return QueryResponse(**response_data)


# ── POST /ingest ──────────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest_endpoint(file: UploadFile = File(...)):
    """
    Upload a .txt or .pdf file to ingest into the FAISS knowledge base.
    Updates the retriever in-place after ingestion.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Only .txt and .pdf are accepted."
        )

    # Save uploaded file to sample_docs/
    docs_dir = Path(settings.sample_docs_path)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / file.filename

    content = await file.read()
    dest.write_bytes(content)
    logger.info(f"Saved uploaded file: {dest} ({len(content)} bytes)")

    try:
        result = ingest_documents(
            docs_path=str(docs_dir),
            index_path=settings.faiss_index_path,
            embedding_model_name=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        # Reload retriever with new index
        reload_retriever(settings.faiss_index_path, settings.embedding_model)
        
        # Flush the cache because the knowledge base has changed
        get_cache(settings.redis_url, settings.cache_ttl).flush_all()
        
        logger.info(f"Ingestion result: {result}")
        return IngestResponse(filename=file.filename, **result)
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── GET /metrics ──────────────────────────────────────────────────────────────
@app.get("/metrics", tags=["Observability"])
async def metrics_endpoint():
    """Platform-level performance and usage metrics."""
    return get_metrics()


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_endpoint():
    """System health: Redis connection, FAISS index, LangSmith status."""
    cache = get_cache(settings.redis_url, settings.cache_ttl)
    retriever = get_retriever(settings.faiss_index_path, settings.embedding_model)
    tracer = get_tracer()

    return HealthResponse(
        status="ok",
        redis=cache.is_available(),
        faiss_loaded=retriever.is_loaded,
        faiss_chunks=retriever.chunk_count,
        langsmith_enabled=tracer.enabled,
    )


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "AutoFabric",
        "version": "1.0.0",
        "description": "Enterprise AI Agent Orchestration Platform",
        "docs": "/docs",
    }
