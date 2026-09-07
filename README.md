# Math Definition Assistant RAG Baseline

This is the runnable RAG baseline program that serves as the non-agentic program for the adaptive and dynamic context-engineering for Agentic RAG capstone. It answers conceptual math and statistics questions from a local corpus of definitions, formulas, comparisons, and examples.

```text
User question
    -> Chroma sentence-transformer embedding
    -> Chroma cosine vector search
    -> retrieve fixed top 5 semantic chunks
    -> question + all 5 chunks sent to Groq once
    -> cited answer
```

The retrieval strategy is intentionally fixed ad relies on a constant value assigned to the `top-k`. The baseline never evaluates context sufficiency, filters or reranks chunks, rewrites a query, changes *k*, or retrieves again.

## Project structure

- `chunking.py` loads and validates the JSON records. Every record is already a semantic chunk, so the loader preserves its definition/formula/example boundary.
- `vector_store.py` creates the persistent Chroma collection and uses `collection.add()` to embed and store chunks.
- `retrieval.py` uses `collection.query()` with fixed `TOP_K = 5`.
- `generator.py` formats all five retrieved chunks and sends one chat-completion request to Groq.
- `baseline_rag.py` is the terminal application and index-rebuild command.
- `data/math_reference.json` is the version-controlled source corpus.

Each Chroma record stores an ID, document text, metadata (`term_id`, `term`, `type`, `title`, and source), and an embedding generated internally by Chroma. The `type` metadata distinguishes a basic definition from a mathematical formula, example, comparison, or related fact.

## Setup

Requirements: Python 3.10 or later, and a Groq API key.

```powershell
cd path\to\cse_598_capstone_proposal

python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Copy the contents in `.env.example` to a new file `.env`
Copy-Item .env.example .env
```
To get a groq API key, go to the groq site: https://groq.com/.
Go to "Start Building" and create an account. Click on API Keys" in the navbar and clicko n the "+ Create API Key" button to create an api key.
NOTE: This API Key will only be visible once. Make sure to save this API Key somewhere in your local device.    
Put your actual Groq key in `.env`:

```text
GROQ_API_KEY=your_actual_key_here
```

Do not commit `.env`, `.venv`, or `chroma_db`; `.gitignore` excludes them.

## Run the baseline

Build or rebuild the local vector database from the committed JSON corpus:

```powershell
.\.venv\Scripts\python.exe baseline_rag.py --rebuild
```

Then run the interactive assistant:

```powershell
.\.venv\Scripts\python.exe baseline_rag.py
```

Example question:

```text
When should I use the median instead of the mean?
```

You may also pass the question directly:

```powershell
.\.venv\Scripts\python.exe baseline_rag.py "What is conditional probability?"
```

## Alternative Version of Running the baseline (More clear):
After you created the virtual envrionrnment, you can activate it. This allows you to access the virtual envriornment:
```powershell
.\.venv\Scripts\Activate.ps1 
```

Then Build or rebuild the local vector  database from the committed JSON Corpus:
```powershell
python baseline_rag.py --rebuild
```

Run the interactive asssistant:
```powershell
python baseline_rag.py
```
Example question. Include this when prompted for a question:

```text
When should I use the median instead of the mean?
```
You may also pass the question directly:

```powershell
python baseline_rag.py "What is conditional probability?"
```




The first Chroma run downloads the local `all-MiniLM-L6-v2` sentence-transformer embedding model. Later runs reuse its cache and the persisted `chroma_db` index.

## Testability and reproducibility

The corpus, chunk IDs, metadata schema, model names, fixed top-*k* value, and dependencies are all version controlled. Rebuilding the index always starts from `data/math_reference.json`.

Run the offline tests without an API key or network call to Groq:

```powershell
.\.venv\Scripts\python.exe -m pytest test_baseline.py
```

The tests verify that the corpus loads as valid semantic chunks, that the retrieval system returns the fixed number of chunks, and that the retrieved chunks are placed in the LLM prompt. The documented concrete example above demonstrates the full retrieval-and-answer pipeline.

## Proposal-ready baseline description

The baseline uses Python, ChromaDB, Chroma's local `all-MiniLM-L6-v2` sentence-transformer embedding function, and the Groq Python SDK with `openai/gpt-oss-120b` for answer generation. It loads a local JSON corpus of 30 semantic math-definition chunks, embeds and persists them in a Chroma cosine-similarity collection, and uses `collection.query()` to retrieve exactly five chunks for every question. The original question and all five chunks are sent to Groq in one generation request, and the response cites chunk IDs.

This is a reasonable starting point because it represents conventional fixed-context RAG while excluding the capstone's proposed contribution. Later, the adaptive agent can assess relevance and context sufficiency, filter or expand context, construct and optimize the context, and retrieve again. Since this baseline always uses the same corpus, embedding model, LLM, and fixed top-*k* retrieval, later performance changes can be attributed to those adaptive context-engineering decisions.