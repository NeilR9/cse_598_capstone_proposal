"""Runnable fixed top-5 Chroma + Groq RAG baseline."""

from __future__ import annotations

import argparse

from chunking import load_and_create_chunks
from generator import generate_answer
from retrieval import retrieve
from vector_store import get_collection, rebuild_collection


def ensure_index() -> None:
    """Build the local collection on first use; later runs reuse the persisted index."""
    if get_collection().count() == 0:
        print("Building local Chroma collection for the first time...")
        rebuild_collection(load_and_create_chunks())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed top-5 math definition RAG baseline")
    parser.add_argument("question", nargs="?", help="Optional question; prompts interactively if omitted.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Chroma vectors from the JSON corpus.")
    args = parser.parse_args()

    if args.rebuild:
        rebuild_collection(load_and_create_chunks())
        if not args.question:
            return
    else:
        ensure_index()

    question = args.question or input("Ask a math/statistics definition question: ").strip()
    if not question:
        parser.error("Please enter a non-empty question.")

    results = retrieve(question)
    print("\nRetrieved chunks (fixed top 5):")
    for chunk_id, metadata, distance in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"- {chunk_id} | distance={distance:.3f} | {metadata['title']}")
    print(f"\nAnswer:\n{generate_answer(question, results)}")


if __name__ == "__main__":
    main()
