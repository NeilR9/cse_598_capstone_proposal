from chunking import load_and_create_chunks
from config import TOP_K
from generator import build_prompt, generate_answer
from retrieval import retrieve


def test_new_math_corpus_loads_as_unique_semantic_chunks():
    chunks = load_and_create_chunks()
    assert len(chunks) >= TOP_K
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert {"definition", "mathematical", "example"} <= {chunk["type"] for chunk in chunks}


def test_math_corpus_contains_expected_chunk_types():
    """The corpus should contain the main semantic chunk types."""
    chunks = load_and_create_chunks()
    chunk_types = {chunk["type"] for chunk in chunks}

    assert {"definition", "mathematical", "example"} <= chunk_types


def test_retrieval_returns_fixed_top_k():
    """The baseline should always retrieve exactly TOP_K chunks."""
    results = retrieve("What is the mean?")

    assert len(results["ids"][0]) == TOP_K
    assert len(results["documents"][0]) == TOP_K
    assert len(results["metadatas"][0]) == TOP_K
    assert len(results["distances"][0]) == TOP_K


def test_prompt_contains_every_retrieved_chunk():
    """Every retrieved chunk should be included in the Groq prompt."""
    results = retrieve("What is the mean?")
    prompt = build_prompt("What is the mean?", results)

    for chunk_id in results["ids"][0]:
        assert f"[{chunk_id}]" in prompt


def test_prompt_contains_fixed_top_k_context():
    """The prompt should contain exactly the fixed TOP_K retrieved chunks."""
    results = retrieve("What is the mean?")
    prompt = build_prompt("What is the mean?", results)

    assert len(results["ids"][0]) == TOP_K

    for chunk_id, document in zip(
        results["ids"][0],
        results["documents"][0],
    ):
        assert f"[{chunk_id}]" in prompt
        assert document in prompt


def test_generate_answer_calls_groq():
    """The generator should send the question and retrieved context to Groq."""

    class FakeMessage:
        content = "The mean is the average of a set of values."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeResponse()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    client = FakeClient()

    results = retrieve("What is the mean?")
    answer = generate_answer(
        "What is the mean?",
        results,
        client=client,
    )

    assert answer == "The mean is the average of a set of values."

    request = client.chat.completions.kwargs

    assert "messages" in request
    assert "model" in request
    assert request["temperature"] == 0

    prompt = request["messages"][1]["content"]

    for chunk_id in results["ids"][0]:
        assert f"[{chunk_id}]" in prompt


def test_definition_question_retrieves_mean_context():
    """A definition question should retrieve context about the mean."""
    results = retrieve("What is the mean?")

    retrieved_terms = {
        metadata["term"]
        for metadata in results["metadatas"][0]
    }

    assert "Mean" in retrieved_terms


def test_relationship_question_retrieves_related_concepts():
    """A relationship question should retrieve multiple related concepts."""
    question = (
        "How does Bayes' rule relate to conditional probability "
        "and independent events?"
    )

    results = retrieve(question)

    retrieved_terms = {
        metadata["term"]
        for metadata in results["metadatas"][0]
    }

    assert "Bayes' Rule" in retrieved_terms
    assert "Conditional Probability" in retrieved_terms
    assert "Independent Events" in retrieved_terms
