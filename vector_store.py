"""Create, populate, and access the persistent Chroma vector collection."""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL

_embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    """Return the persistent collection with cosine-distance search enabled."""
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def rebuild_collection(chunks: list[dict[str, str]]):
    """Replace the local index with vectors derived from the supplied chunks."""
    try:
        _client.delete_collection(CHROMA_COLLECTION)
    except ValueError:
        pass  # No prior local collection on the first run.
    collection = get_collection()
    embed_and_store(chunks, collection)
    return collection


def embed_and_store(chunks: list[dict[str, str]], collection=None) -> None:
    """Store text and metadata; Chroma generates and persists embeddings itself."""
    collection = collection or get_collection()
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "term_id": chunk["term_id"],
                "term": chunk["term"],
                "type": chunk["type"],
                "title": chunk["title"],
                "source": "math_reference.json",
            }
            for chunk in chunks
        ],
    )
    print(f"Stored {len(chunks)} semantic chunks in '{CHROMA_COLLECTION}'.")
