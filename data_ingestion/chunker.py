"""
chunker.py — Split raw page dicts into overlapping chunks using LangChain.

Output chunk dict keys:
    text, faculty, doc_type, source_url | source_file, last_updated, chunk_id
"""

import hashlib
import os
import sys

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)


def _make_chunk_id(text: str, source: str, idx: int) -> str:
    raw = f"{source}::{idx}::{text[:64]}"
    return hashlib.md5(raw.encode()).hexdigest()


def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Split each page dict into overlapping text chunks.

    Args:
        pages: list of page dicts (from scraper, pdf_extractor, or manual loader)
        chunk_size: override config value if provided
        chunk_overlap: override config value if provided

    Returns:
        list of chunk dicts ready for embedding
    """
    size = chunk_size or _cfg["chunking"]["chunk_size"]
    overlap = chunk_overlap or _cfg["chunking"]["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []

    for page in pages:
        raw_text = page.get("text", "").strip()
        if not raw_text:
            continue

        source = page.get("url") or page.get("source_file") or "unknown"

        splits = splitter.split_text(raw_text)
        for idx, split_text in enumerate(splits):
            if len(split_text.strip()) < 30:
                continue
            chunk = {
                "text": split_text.strip(),
                "faculty": page.get("faculty", "General"),
                "doc_type": page.get("doc_type", "general"),
                "language": page.get("language", "ru"),
                "last_updated": page.get("last_updated", ""),
                "chunk_id": _make_chunk_id(split_text, source, idx),
            }
            if "url" in page:
                chunk["source_url"] = page["url"]
            if "source_file" in page:
                chunk["source_file"] = page["source_file"]
                chunk["page"] = page.get("page", 0)
            chunks.append(chunk)

    print(f"[chunker] {len(pages)} pages → {len(chunks)} chunks "
          f"(size={size}, overlap={overlap})")
    return chunks


if __name__ == "__main__":
    dummy = [{"text": "Lorem ipsum dolor sit amet. " * 100,
              "url": "http://example.com", "faculty": "CS",
              "doc_type": "program", "last_updated": "2026-05-30"}]
    result = chunk_pages(dummy)
    print(f"Chunks produced: {len(result)}")
    print("First chunk preview:", result[0]["text"][:80])
