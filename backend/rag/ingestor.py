"""
AutoFabric RAG — Document Ingestion Pipeline
Reads .txt and .pdf files, chunks them, embeds with sentence-transformers,
stores in FAISS (IndexFlatL2), and saves to disk alongside a BM25 corpus.
"""
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ── Lazy import for PDF support ──────────────────────────────────────────────
def _try_import_pypdf():
    try:
        from pypdf import PdfReader
        return PdfReader
    except ImportError:
        logger.warning("pypdf not installed — PDF ingestion disabled")
        return None


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Simple word-count based chunker with overlap.
    Splits on whitespace tokens to approximate token count.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    PdfReader = _try_import_pypdf()
    if PdfReader is None:
        return ""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def ingest_documents(
    docs_path: str,
    index_path: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Dict:
    """
    Full ingestion pipeline:
    1. Reads .txt / .pdf files from docs_path
    2. Chunks each document
    3. Embeds chunks with sentence-transformers
    4. Builds / updates FAISS IndexFlatL2
    5. Persists index + raw chunks + metadata to disk

    Returns: {status, chunks_added, total_chunks}
    """
    docs_dir = Path(docs_path)
    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ── Paths ─────────────────────────────────────────────────────────────────
    faiss_file = index_dir / "index.faiss"
    chunks_file = index_dir / "chunks.pkl"
    meta_file = index_dir / "meta.json"

    # ── Load existing state if available ─────────────────────────────────────
    existing_chunks: List[Dict] = []
    existing_index = None
    if chunks_file.exists() and faiss_file.exists():
        with open(chunks_file, "rb") as f:
            existing_chunks = pickle.load(f)
        existing_index = faiss.read_index(str(faiss_file))
        logger.info(f"Loaded existing index with {len(existing_chunks)} chunks")

    # ── Load embedding model ──────────────────────────────────────────────────
    logger.info(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    dim = model.get_sentence_embedding_dimension()

    # ── Discover files ────────────────────────────────────────────────────────
    supported = [".txt", ".pdf"]
    files = [f for f in docs_dir.rglob("*") if f.suffix.lower() in supported]
    logger.info(f"Found {len(files)} document(s) in {docs_dir}")

    # ── Track already-ingested sources ────────────────────────────────────────
    ingested_sources = {c["source"] for c in existing_chunks}

    new_chunks: List[Dict] = []
    for file_path in files:
        source_name = file_path.name
        if source_name in ingested_sources:
            logger.info(f"Skipping already-ingested: {source_name}")
            continue

        logger.info(f"Ingesting: {source_name}")
        if file_path.suffix.lower() == ".txt":
            raw_text = _read_txt(file_path)
        else:
            raw_text = _read_pdf(file_path)

        if not raw_text.strip():
            logger.warning(f"Empty content in {source_name}, skipping")
            continue

        pieces = _chunk_text(raw_text, chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            new_chunks.append({
                "content": piece,
                "source": source_name,
                "chunk_id": f"{source_name}_{i}",
            })

    if not new_chunks:
        total = len(existing_chunks)
        logger.info("No new documents to ingest")
        return {"status": "no_new_documents", "chunks_added": 0, "total_chunks": total}

    # ── Embed new chunks ──────────────────────────────────────────────────────
    logger.info(f"Embedding {len(new_chunks)} new chunks...")
    texts = [c["content"] for c in new_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # ── Build / extend FAISS index ────────────────────────────────────────────
    if existing_index is None:
        index = faiss.IndexFlatL2(dim)
    else:
        index = existing_index

    index.add(embeddings)

    # ── Merge chunks ──────────────────────────────────────────────────────────
    all_chunks = existing_chunks + new_chunks

    # ── Persist ───────────────────────────────────────────────────────────────
    faiss.write_index(index, str(faiss_file))
    with open(chunks_file, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(meta_file, "w") as f:
        json.dump({"total_chunks": len(all_chunks), "dim": dim}, f)

    logger.info(f"Ingestion complete. Total chunks: {len(all_chunks)}")
    return {
        "status": "success",
        "chunks_added": len(new_chunks),
        "total_chunks": len(all_chunks),
    }


def get_index_stats(index_path: str) -> Dict:
    """Return stats about the current FAISS index."""
    meta_file = Path(index_path) / "meta.json"
    if not meta_file.exists():
        return {"total_chunks": 0, "dim": 0, "loaded": False}
    with open(meta_file) as f:
        meta = json.load(f)
    meta["loaded"] = True
    return meta
