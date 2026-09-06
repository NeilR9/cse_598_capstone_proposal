"""Fixed-depth vector retrieval for the baseline RAG system."""

from __future__ import annotations

from config import TOP_K
from vector_store import get_collection


def retrieve(question: str):
    """Return exactly the fixed top-k nearest semantic chunks for a question."""
    if not question or not question.strip():
        raise ValueError("A non-empty question is required.")
    collection = get_collection()
    if collection.count() < TOP_K:
        raise RuntimeError(
            f"Collection has fewer than {TOP_K} records. Run baseline_rag.py --rebuild first."
        )
    return collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
