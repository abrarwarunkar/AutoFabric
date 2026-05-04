"""
AutoFabric — MCP Server
Exposes 3 standardized tools via the official MCP Python SDK over stdio transport.

Tools:
  1. search_tool   — Hybrid RAG retrieval (FAISS + BM25 + reranker)
  2. summarize_tool — LLaMA 3.1 8B summarization via Groq
  3. validate_tool  — Hallucination grounding check via Groq

Run standalone:
    python -m backend.mcp.mcp_server
"""
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure backend/ is importable when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from groq import Groq

from backend.config import settings
from backend.rag.retriever import get_retriever
from backend.rag.reranker import get_reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── MCP Server Init ───────────────────────────────────────────────────────────
app = Server("autofabric-mcp")
groq_client = Groq(api_key=settings.groq_api_key)


# ── Helper: Groq call ─────────────────────────────────────────────────────────
def _groq_chat(system_prompt: str, user_message: str, model: str = None) -> str:
    model = model or settings.groq_model_fast
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── Tool Implementations ──────────────────────────────────────────────────────

def _search(query: str, top_k: int = 5):
    """Internal: hybrid retrieval → reranking."""
    retriever = get_retriever(
        index_path=settings.faiss_index_path,
        embedding_model=settings.embedding_model,
    )
    reranker = get_reranker(model_name=settings.reranker_model)

    if not retriever.is_loaded:
        return []

    raw_results = retriever.retrieve(query, top_k=top_k * 2)
    if not raw_results:
        return []

    reranked = reranker.rerank(query, raw_results, top_k=top_k)
    return [
        {
            "content": r["content"],
            "source": r["source"],
            "score": r.get("cross_encoder_score", r.get("rrf_score", 0.0)),
        }
        for r in reranked
    ]


def _summarize(context: str, query: str) -> str:
    """Internal: LLaMA 3.1 8B summarization."""
    system = (
        "You are a precise research assistant. "
        "Given a context and a query, write a concise, accurate summary "
        "that answers the query using ONLY information from the context. "
        "Maximum 200 words. No preamble."
    )
    user = f"QUERY: {query}\n\nCONTEXT:\n{context[:4000]}"
    return _groq_chat(system, user, model=settings.groq_model_fast)


def _validate(response: str, context: str) -> dict:
    """Internal: grounding validation via LLaMA 3.1 8B."""
    system = (
        "You are a factual grounding validator. "
        "Check if the RESPONSE is fully supported by the CONTEXT. "
        "Reply with a JSON object only: "
        '{"is_grounded": bool, "confidence": float 0-1, "flags": [list of unsupported claims]}'
    )
    user = f"RESPONSE:\n{response}\n\nCONTEXT:\n{context[:3000]}"
    raw = _groq_chat(system, user, model=settings.groq_model_fast)

    # Parse JSON response
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"Validate tool returned non-JSON: {raw[:100]}")
        return {"is_grounded": False, "confidence": 0.0, "flags": ["parse_error"]}


# ── MCP Tool Handlers ─────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_tool",
            description="Hybrid semantic + BM25 search over the AutoFabric knowledge base. Returns top-k grounded results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="summarize_tool",
            description="Summarize a context passage in answer to a query. Uses LLaMA 3.1 8B.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Source context to summarize"},
                    "query": {"type": "string", "description": "The question to answer"},
                },
                "required": ["context", "query"],
            },
        ),
        types.Tool(
            name="validate_tool",
            description="Validate that a response is grounded in the provided context. Returns is_grounded, confidence, flags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "response": {"type": "string", "description": "The response to validate"},
                    "context": {"type": "string", "description": "The source context"},
                },
                "required": ["response", "context"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    logger.info(f"MCP tool called: {name}")

    if name == "search_tool":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        results = await asyncio.get_event_loop().run_in_executor(
            None, _search, query, top_k
        )
        return [types.TextContent(type="text", text=json.dumps(results))]

    elif name == "summarize_tool":
        context = arguments["context"]
        query = arguments["query"]
        summary = await asyncio.get_event_loop().run_in_executor(
            None, _summarize, context, query
        )
        return [types.TextContent(type="text", text=summary)]

    elif name == "validate_tool":
        response = arguments["response"]
        context = arguments["context"]
        result = await asyncio.get_event_loop().run_in_executor(
            None, _validate, response, context
        )
        return [types.TextContent(type="text", text=json.dumps(result))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ── Entrypoint ────────────────────────────────────────────────────────────────
async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_run())


# ── Direct-call wrappers (used by agents without subprocess) ──────────────────
def mcp_search(query: str, top_k: int = 5):
    return _search(query, top_k)


def mcp_summarize(context: str, query: str) -> str:
    return _summarize(context, query)


def mcp_validate(response: str, context: str) -> dict:
    return _validate(response, context)
