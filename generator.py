"""Generate grounded answers from fixed retrieved context using Groq."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

from config import GROQ_MODEL


def build_prompt(question: str, results: dict) -> str:
    """Format all retrieved chunks for one Groq generation call."""
    context_blocks = []
    for chunk_id, document, metadata in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    ):
        context_blocks.append(
            f"[{chunk_id}] {metadata['title']} ({metadata['type']}):\n{document}"
        )
    context = "\n\n".join(context_blocks)
    return f"""Answer the math/statistics question using only the reference chunks.
Give a concise, accurate explanation and cite source chunk IDs in brackets.
If the chunks do not support the answer, say so clearly.

Question: {question}

Reference chunks:
{context}"""


def generate_answer(question: str, results: dict, client: Groq | None = None) -> str:
    """Send the question and all fixed top-k chunks to Groq once."""
    load_dotenv()
    if client is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or the shell environment.")
        client = Groq()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a careful math and statistics definition assistant."},
            {"role": "user", "content": build_prompt(question, results)},
        ],
    )
    return response.choices[0].message.content or "No answer was returned."
