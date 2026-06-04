"""
embedder.py — Embed text chunks with BGE-m3 and store in ChromaDB.

Run directly to ingest all PDF + manual text data:
    python data_ingestion/embedder.py
"""

import json
import os
import sys
from datetime import date
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

# Resolve config paths relative to project root, not CWD
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _abs(rel_path: str) -> str:
    """Make a config path absolute regardless of CWD."""
    p = Path(rel_path)
    return str(p if p.is_absolute() else _PROJECT_ROOT / p)


from data_ingestion.pdf_extractor import extract_all_pdfs
from data_ingestion.chunker import chunk_pages


# ── Singleton model ────────────────────────────────────────────────────────────

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = _cfg["embeddings"]["dense_model"]
        device = _cfg["embeddings"]["device"]
        print(f"[embedder] Loading {model_name} on {device}…")
        _embedding_model = SentenceTransformer(model_name, device=device)
    return _embedding_model


# ── ChromaDB client ────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_abs(_cfg["chroma"]["persist_dir"]))
    collection = client.get_or_create_collection(
        name=_cfg["chroma"]["collection_name"],
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Manual .txt file loader ────────────────────────────────────────────────────

_FACULTY_FROM_FILENAME: dict[str, str] = {
    "engineering": "Факультет инженерии и информатики",
    "economics": "Факультет экономики и управления",
    "medicine": "Медицинский факультет",
    "humanities": "Факультет гуманитарных наук",
    "social": "Факультет социальных наук",
    "admissions": "General",
    "ort": "General",
    "faq": "General",
    "contact": "General",
}


def _infer_faculty_from_txt(filename: str) -> str:
    name = Path(filename).stem.lower()
    for keyword, faculty in _FACULTY_FROM_FILENAME.items():
        if keyword in name:
            return faculty
    return "General"


def load_manual_pages(manual_dir: str | None = None) -> list[dict]:
    """
    Read .txt files from data/raw/manual/{ru,ky,en}/ subdirs.
    Each subdirectory name becomes the `language` metadata field on every chunk.
    Falls back to scanning the base dir directly (tags as 'ru') for old flat layouts.
    """
    base_dir = Path(manual_dir) if manual_dir else Path(_abs(_cfg["data"]["raw_manual"]))
    if not base_dir.exists():
        print(f"[embedder] Manual dir not found: {base_dir}. Skipping.")
        return []

    # Collect (directory, language_code) pairs
    lang_dirs: list[tuple[Path, str]] = [
        (base_dir / lang, lang)
        for lang in ("ru", "ky", "en")
        if (base_dir / lang).is_dir()
    ]
    if not lang_dirs:
        # Legacy flat layout — treat everything as Russian
        lang_dirs = [(base_dir, "ru")]

    pages: list[dict] = []
    today = str(date.today())

    for lang_dir, lang in lang_dirs:
        txt_files = sorted(lang_dir.glob("*.txt"))
        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8").strip()
            if len(text) < 50:
                continue
            pages.append({
                "text": text,
                "faculty": _infer_faculty_from_txt(txt_file.name),
                "doc_type": "manual",
                "source_file": txt_file.name,
                "language": lang,
                "page": 0,
                "last_updated": today,
            })
            print(f"[embedder] [{lang}] {txt_file.name} ({len(text)} chars)")

    print(f"[embedder] {len(pages)} manual text files loaded.")
    return pages


# ── Upsert chunks ──────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict], batch_size: int = 64) -> int:
    """Embed and upsert chunks into ChromaDB. Chunk IDs are deterministic — safe to re-run."""
    if not chunks:
        return 0

    model = get_embedding_model()
    collection = get_collection()

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    metadatas = []
    for c in chunks:
        metadatas.append({
            "faculty": c.get("faculty", "General"),
            "doc_type": c.get("doc_type", "general"),
            "language": c.get("language", "ru"),
            "last_updated": c.get("last_updated", ""),
            "source_url": c.get("source_url", ""),
            "source_file": c.get("source_file", ""),
            "page": int(c.get("page", 0)),
        })

    total = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]
        batch_meta = metadatas[i: i + batch_size]

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=_cfg["embeddings"]["normalize"],
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
    n_results: int | None = None,
    where: dict | None = None,
) -> list[dict]:
    """
    Semantic search over ChromaDB.

    Args:
        query_text: user query (any of RU/KG/EN)
        n_results: number of results (defaults to config retrieval.top_k)
        where: optional ChromaDB metadata filter, e.g. {"faculty": "Медицинский факультет"}

    Returns:
        list of dicts with keys: text, faculty, doc_type, source_url/source_file, distance
    """
    k = n_results or _cfg["retrieval"]["top_k"]
    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=_cfg["embeddings"]["normalize"],
    ).tolist()

    kwargs: dict = {"query_embeddings": query_embedding, "n_results": k}
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
            "source_file": meta.get("source_file", ""),
            "distance": round(dist, 4),
        })
    return output


# ── Full ingestion pipeline ────────────────────────────────────────────────────

def run_full_ingestion() -> None:
    """Manual .txt files + PDFs → chunk → embed → store in ChromaDB."""
    print("=" * 60)

    print("PHASE 1: Loading manual text files…")
    manual_pages = load_manual_pages()

    print("PHASE 2: Extracting PDFs…")
    pdf_pages = extract_all_pdfs()

    all_pages = manual_pages + pdf_pages
    print(f"PHASE 3: Chunking {len(all_pages)} pages…")
    chunks = chunk_pages(all_pages)

    print(f"PHASE 4: Embedding & upserting {len(chunks)} chunks…")
    upserted = upsert_chunks(chunks)
    print(f"✓ Ingestion complete. {upserted} chunks in ChromaDB.")
    print("=" * 60)


if __name__ == "__main__":
    run_full_ingestion()
