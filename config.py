"""Central configuration for the fixed-top-k RAG baseline."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "math_reference.json"
CHROMA_PATH = str(PROJECT_DIR / "chroma_db")
CHROMA_COLLECTION = "math_concept_definitions"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5  # Fixed by design: the baseline never adapts retrieval depth.
