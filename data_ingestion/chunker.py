"""
chunker.py — Split raw page dicts into overlapping chunks using LangChain.

Output chunk dict keys:
    text, faculty, doc_type, source_url | source_file, last_updated, chunk_id
"""

import hashlib
import sys
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _make_chunk_id(text: str, source: str, idx: int) -> str:
    """Deterministic ID so re-ingestion overwrites the same chunk."""
    raw = f"{source}::{idx}::{text[:64]}"
    return hashlib.md5(raw.encode()).hexdigest()


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split each page dict into overlapping text chunks.

    Args:
        pages: list of page dicts (from scraper or pdf_extractor)
        chunk_size: max tokens per chunk (treated as characters here;
                    true token count varies by model)
        chunk_overlap: overlap between consecutive chunks

    Returns:
        list of chunk dicts ready for embedding
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []

    for page in pages:
        raw_text = page.get("text", "").strip()
        if not raw_text:
            continue

        # Determine a stable source key for chunk_id generation
        source = page.get("url") or page.get("source_file") or "unknown"

        splits = splitter.split_text(raw_text)
        for idx, split_text in enumerate(splits):
            if len(split_text.strip()) < 30:
                continue
            chunk = {
                "text": split_text.strip(),
                "faculty": page.get("faculty", "General"),
                "doc_type": page.get("doc_type", "general"),
                "last_updated": page.get("last_updated", ""),
                "chunk_id": _make_chunk_id(split_text, source, idx),
            }
            # Preserve the correct source key
            if "url" in page:
                chunk["source_url"] = page["url"]
            if "source_file" in page:
                chunk["source_file"] = page["source_file"]
                chunk["page"] = page.get("page", 0)
            chunks.append(chunk)

    print(f"[chunker] {len(pages)} pages → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


if __name__ == "__main__":
    # Quick smoke test with dummy data
    dummy = [{"text": "Lorem ipsum dolor sit amet. " * 100,
              "url": "http://example.com", "faculty": "CS",
              "doc_type": "program", "last_updated": "2026-04-13"}]
    result = chunk_pages(dummy)
    print(f"Chunks produced: {len(result)}")
    print("First chunk preview:", result[0]["text"][:80])
