"""
embedder.py — Embed text chunks with BGE-m3 and store in ChromaDB.

Run directly to ingest all scraped + PDF data:
    python data_ingestion/embedder.py
"""

import json
import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL, EMBEDDING_DEVICE,
    ORT_THRESHOLDS_FILE, DATA_DIR,
)
from data_ingestion.scraper import scrape_all
from data_ingestion.pdf_extractor import extract_all_pdfs
from data_ingestion.chunker import chunk_pages


# ── Singleton model (load once per process) ────────────────────────────────────

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[embedder] Loading {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}…")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _embedding_model


# ── ChromaDB client ────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Upsert chunks ──────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict], batch_size: int = 64) -> int:
    """
    Embed and upsert chunks into ChromaDB.
    Uses chunk_id as document ID so re-runs overwrite stale embeddings.

    Returns number of chunks upserted.
    """
    if not chunks:
        return 0

    model = get_embedding_model()
    collection = get_collection()

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    # Build metadata (ChromaDB requires all values to be str/int/float/bool)
    metadatas = []
    for c in chunks:
        meta = {
            "faculty": c.get("faculty", "General"),
            "doc_type": c.get("doc_type", "general"),
            "last_updated": c.get("last_updated", ""),
            "source_url": c.get("source_url", ""),
            "source_file": c.get("source_file", ""),
            "page": int(c.get("page", 0)),
        }
        metadatas.append(meta)

    total = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]
        batch_meta = metadatas[i: i + batch_size]

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        total += len(batch_texts)
        print(f"[embedder] Upserted {total}/{len(texts)} chunks…")

    print(f"[embedder] Done. Collection size: {collection.count()}")
    return total


# ── Query helper (used by agent tools) ────────────────────────────────────────

def query_collection(
    query_text: str,
    n_results: int = 3,
    where: dict | None = None,
) -> list[dict]:
    """
    Semantic search over ChromaDB.

    Args:
        query_text: user query (any of RU/KG/EN)
        n_results: number of results to return
        where: optional ChromaDB metadata filter, e.g. {"faculty": "CS"}

    Returns:
        list of dicts with keys: text, faculty, doc_type, source_url, distance
    """
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode(
        [query_text], normalize_embeddings=True
    ).tolist()

    kwargs: dict = {"query_embeddings": query_embedding, "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "text": doc,
            "faculty": meta.get("faculty", ""),
            "doc_type": meta.get("doc_type", ""),
            "source_url": meta.get("source_url", ""),
            "distance": round(dist, 4),
        })
    return output


# ── Full ingestion pipeline ────────────────────────────────────────────────────

def run_full_ingestion() -> None:
    """Scrape website + extract PDFs → chunk → embed → store."""
    print("=" * 60)
    print("PHASE 1: Scraping university website…")
    web_pages, ort_thresholds = scrape_all()

    # Persist ORT thresholds if any were found
    if ort_thresholds:
        existing: dict = {}
        if os.path.exists(ORT_THRESHOLDS_FILE):
            with open(ORT_THRESHOLDS_FILE) as f:
                existing = json.load(f)
        existing.update(ort_thresholds)
        with open(ORT_THRESHOLDS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[embedder] ORT thresholds saved → {ORT_THRESHOLDS_FILE}")

    print("PHASE 2: Extracting PDFs…")
    pdf_pages = extract_all_pdfs()

    all_pages = web_pages + pdf_pages
    print(f"PHASE 3: Chunking {len(all_pages)} pages…")
    chunks = chunk_pages(all_pages)

    print(f"PHASE 4: Embedding & upserting {len(chunks)} chunks…")
    upserted = upsert_chunks(chunks)
    print(f"✓ Ingestion complete. {upserted} chunks in ChromaDB.")
    print("=" * 60)


if __name__ == "__main__":
    run_full_ingestion()
