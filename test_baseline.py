"""Offline tests for corpus loading and fixed-context prompt construction."""

from chunking import load_and_create_chunks
from config import TOP_K
from generator import build_prompt


def test_new_math_corpus_loads_as_unique_semantic_chunks():
    chunks = load_and_create_chunks()
    assert len(chunks) >= TOP_K
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert {"definition", "mathematical", "example"} <= {chunk["type"] for chunk in chunks}


def test_prompt_contains_every_fixed_retrieved_chunk():
    chunks = load_and_create_chunks()[:TOP_K]
    results = {
        "ids": [[chunk["id"] for chunk in chunks]],
        "documents": [[chunk["text"] for chunk in chunks]],
        "metadatas": [[
            {"title": chunk["title"], "type": chunk["type"]} for chunk in chunks
        ]],
    }
    prompt = build_prompt("What is a mean?", results)
    for chunk in chunks:
        assert f"[{chunk['id']}]" in prompt
    assert len(results["ids"][0]) == TOP_K
