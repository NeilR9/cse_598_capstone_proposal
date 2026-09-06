"""Load and validate semantic chunks for the math definition assistant."""

from __future__ import annotations

import json
from pathlib import Path

from config import DATA_PATH

REQUIRED_FIELDS = {"id", "term_id", "term", "type", "title", "text"}


def load_and_create_chunks(data_path: Path = DATA_PATH) -> list[dict[str, str]]:
    """Load the pre-chunked concept records from the JSON corpus.

    Each record is already a semantic chunk: one concept's definition, formula,
    comparison, or example. This avoids splitting a formula from its meaning.
    """
    with data_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("The dataset must be a non-empty JSON list.")

    chunks: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Chunk is missing required fields: {sorted(missing)}")
        if record["id"] in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {record['id']}")
        if not record["text"].strip():
            raise ValueError(f"Chunk {record['id']} has no text.")
        seen_ids.add(record["id"])
        chunks.append({field: str(record[field]).strip() for field in REQUIRED_FIELDS})
    return chunks
