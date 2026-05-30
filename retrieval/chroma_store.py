"""
chroma_store.py — Hybrid BM25 + ChromaDB EnsembleRetriever.

BM25 handles exact Kyrgyz/Russian term matching (e.g. "ОРТ упайы", "кабылуу").
BGE-m3 dense embeddings handle semantic meaning across all three languages.
30/70 weight split as per CLAUDE.md ablation configuration.
"""

import os
import sys
from typing import Optional

import yaml
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_dense_embeddings: HuggingFaceEmbeddings | None = None


def load_embeddings(use_ablation: bool = False) -> HuggingFaceEmbeddings:
    """Load BGE-m3 (default) or multilingual-e5-large (ablation baseline)."""
    global _dense_embeddings
    if _dense_embeddings is None or use_ablation:
        model = (
            _cfg["embeddings"]["ablation_model"]
            if use_ablation
            else _cfg["embeddings"]["dense_model"]
        )
        _dense_embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": _cfg["embeddings"]["device"]},
            encode_kwargs={"normalize_embeddings": _cfg["embeddings"]["normalize"]},
        )
    return _dense_embeddings


def get_retriever(
    docs: list[Document],
    faculty_filter: Optional[str] = None,
    use_ablation: bool = False,
) -> EnsembleRetriever:
    """
    Build a hybrid BM25 + ChromaDB EnsembleRetriever.

    Args:
        docs: list of LangChain Documents (all ingested docs for BM25 index)
        faculty_filter: optional faculty name to filter ChromaDB results
        use_ablation: use multilingual-e5-large instead of BGE-m3 (Section 4 comparison)

    Returns:
        EnsembleRetriever with 30% BM25 + 70% dense weights
    """
    top_k = _cfg["retrieval"]["top_k"]

    search_kwargs: dict = {"k": top_k}
    if faculty_filter:
        search_kwargs["filter"] = {"faculty": faculty_filter}

    dense = Chroma(
        collection_name=_cfg["chroma"]["collection_name"],
        embedding_function=load_embeddings(use_ablation),
        persist_directory=_cfg["chroma"]["persist_dir"],
    ).as_retriever(search_kwargs=search_kwargs)

    bm25 = BM25Retriever.from_documents(docs, k=top_k)

    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[
            _cfg["retrieval"]["bm25_weight"],
            _cfg["retrieval"]["dense_weight"],
        ],
    )


def docs_from_collection() -> list[Document]:
    """
    Pull all stored documents from ChromaDB as LangChain Document objects.
    Used to build the BM25 index over the same corpus as the dense retriever.
    """
    import chromadb
    client = chromadb.PersistentClient(path=_cfg["chroma"]["persist_dir"])
    try:
        col = client.get_collection(_cfg["chroma"]["collection_name"])
    except Exception:
        return []

    total = col.count()
    if total == 0:
        return []

    result = col.get(limit=total, include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs
