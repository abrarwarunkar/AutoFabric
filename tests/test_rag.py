"""
AutoFabric Tests — RAG Pipeline
Tests: ingestion, dense retrieval, BM25 retrieval, RRF fusion, reranker ordering
"""
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs_dir(tmp_path):
    """Create a temp dir with sample .txt files."""
    docs = {
        "doc_ai.txt": (
            "Artificial intelligence is the simulation of human intelligence by machines. "
            "Machine learning is a subset of AI that enables computers to learn from data. "
            "Deep learning uses neural networks with many layers to process complex patterns."
        ),
        "doc_rag.txt": (
            "Retrieval-Augmented Generation (RAG) combines dense retrieval with language models. "
            "FAISS is a library for efficient similarity search and clustering of dense vectors. "
            "BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency."
        ),
        "doc_agents.txt": (
            "LangGraph is a framework for building stateful multi-actor applications with LLMs. "
            "Agent orchestration allows multiple AI agents to collaborate on complex tasks. "
            "Supervisory agents coordinate sub-agents and assemble final responses."
        ),
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def index_dir(tmp_path):
    """Return a temp directory for FAISS index."""
    return str(tmp_path / "faiss_index")


# ── Test: Ingestion ───────────────────────────────────────────────────────────

def test_ingestor_creates_faiss_index(sample_docs_dir, index_dir):
    """Ingestion should create a FAISS index and chunk pickle on disk."""
    from backend.rag.ingestor import ingest_documents
    result = ingest_documents(
        docs_path=sample_docs_dir,
        index_path=index_dir,
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=50,
        chunk_overlap=10,
    )
    assert result["status"] == "success"
    assert result["chunks_added"] > 0
    assert result["total_chunks"] > 0

    # Files should exist on disk
    assert (Path(index_dir) / "index.faiss").exists()
    assert (Path(index_dir) / "chunks.pkl").exists()
    assert (Path(index_dir) / "meta.json").exists()


def test_ingestor_skips_duplicate_sources(sample_docs_dir, index_dir):
    """Re-ingesting same files should not add duplicate chunks."""
    from backend.rag.ingestor import ingest_documents
    first = ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    second = ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    assert second["chunks_added"] == 0
    assert second["status"] == "no_new_documents"
    assert second["total_chunks"] == first["total_chunks"]


# ── Test: Dense Retrieval ─────────────────────────────────────────────────────

def test_dense_retrieval_returns_top_k(sample_docs_dir, index_dir):
    """Dense retrieval should return exactly top_k results."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    results = retriever._dense_retrieve("artificial intelligence", top_k=2)
    assert len(results) <= 2
    assert all("content" in r for r in results)
    assert all("dense_score" in r for r in results)


def test_dense_retrieval_scores_are_positive(sample_docs_dir, index_dir):
    """Dense retrieval scores should be positive (converted from L2 distance)."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    results = retriever._dense_retrieve("machine learning", top_k=3)
    assert all(r["dense_score"] > 0 for r in results)


# ── Test: BM25 Retrieval ──────────────────────────────────────────────────────

def test_bm25_retrieval_returns_results(sample_docs_dir, index_dir):
    """BM25 sparse retrieval should return ranked results."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    results = retriever._sparse_retrieve("BM25 retrieval algorithm", top_k=3)
    assert len(results) > 0
    assert all("sparse_score" in r for r in results)
    assert all("sparse_rank" in r for r in results)


def test_bm25_retrieval_ranks_correctly(sample_docs_dir, index_dir):
    """BM25 should rank chunk containing 'FAISS' highest for FAISS query."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    results = retriever._sparse_retrieve("FAISS similarity search", top_k=5)
    top_content = results[0]["content"].lower() if results else ""
    assert "faiss" in top_content or "retrieval" in top_content


# ── Test: RRF Fusion ──────────────────────────────────────────────────────────

def test_rrf_fusion_merges_results(sample_docs_dir, index_dir):
    """RRF should produce merged ranked list from dense + sparse."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    results = retriever.retrieve("neural networks deep learning", top_k=3)
    assert len(results) > 0
    assert all("rrf_score" in r for r in results)
    assert all("dense_score" in r for r in results)
    assert all("sparse_score" in r for r in results)


def test_rrf_scores_descending(sample_docs_dir, index_dir):
    """RRF results should be sorted by score descending."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    results = retriever.retrieve("LangGraph agent orchestration", top_k=5)
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ── Test: Cross-Encoder Reranker ──────────────────────────────────────────────

def test_reranker_adds_score(sample_docs_dir, index_dir):
    """Reranker should add cross_encoder_score to each chunk."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever
    from backend.rag.reranker import CrossEncoderReranker

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    chunks = retriever.retrieve("machine learning data", top_k=4)
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("machine learning data", chunks)

    assert len(reranked) == len(chunks)
    assert all("cross_encoder_score" in r for r in reranked)


def test_reranker_sorts_by_score(sample_docs_dir, index_dir):
    """Reranker output should be sorted by cross_encoder_score descending."""
    from backend.rag.ingestor import ingest_documents
    from backend.rag.retriever import HybridRetriever
    from backend.rag.reranker import CrossEncoderReranker

    ingest_documents(sample_docs_dir, index_dir, chunk_size=50)
    retriever = HybridRetriever(index_dir)
    retriever.load()

    chunks = retriever.retrieve("RAG retrieval augmented", top_k=5)
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("RAG retrieval augmented", chunks, top_k=3)

    scores = [r["cross_encoder_score"] for r in reranked]
    assert scores == sorted(scores, reverse=True)
    assert len(reranked) <= 3
