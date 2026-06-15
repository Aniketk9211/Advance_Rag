# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A LangGraph implementation of **Corrective RAG (CRAG)**: a retrieval pipeline that grades retrieved
documents for relevance, falls back to a Tavily web search when the vector store doesn't have a good
answer, and generates a final response with an LLM (Google Gemini via `langchain-google-genai`).
The project is in early/active development — several modules referenced by other files do not exist
yet, and some existing files have not been run successfully.

## Commands

This project uses `uv` for dependency management (Python 3.11, see `.python-version`).

- Install/sync dependencies: `uv sync`
- Run the app entry point: `uv run python main.py`
- Run all tests: `uv run pytest` (must be run from the project root — see import notes below)
- Run a single test: `uv run pytest tests/test_chain.py::test_router_to_websearch`
- Lint: `uv run ruff check .`
- (Re)build the local vector store: `uv run python ingestion.py` — this scrapes the source URLs,
  chunks them, embeds them with `GoogleGenerativeAIEmbeddings`, and persists to `./.chroma`. It only
  re-ingests if the Chroma collection is currently empty.

Required environment variables live in `.env` (gitignored): `GOOGLE_API_KEY` (LLM + embeddings),
`GROQ_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY` (web search node), and `LANGSMITH_*` (tracing).

## Architecture

### State

`src/graph/state.py` defines `GraphState`, a `TypedDict` threaded through every node of the LangGraph
graph: `question`, `generation`, `web_search`, `documents`. Every node function takes the full state
dict and returns a partial dict of updates to merge back in.

### Ingestion (`ingestion.py`, project root)

Run as a module-level side effect on import: loads three Lilian Weng blog posts via `WebBaseLoader`,
splits them with `RecursiveCharacterTextSplitter`, embeds with `GoogleGenerativeAIEmbeddings`, and
stores them in a Chroma collection (`rag-chroma`) persisted at `./.chroma`. Exposes a single
`retriever` object (`vectorstore.as_retriever()`) that is imported by node/chain modules.

### `src/graph/` package layout

- `node/` — LangGraph node functions, one per pipeline step:
  - `retrieve.py` — vector store retrieval via `retriever.invoke(question)`
  - `grade_documents.py` — LLM-based relevance grading per document; filters out irrelevant docs and
    sets `web_search=True` if any document is judged irrelevant
  - `websearch.py` — Tavily fallback search node (currently incomplete — does not yet return a state
    update)
- `chain/` — LCEL chains (prompt | structured-output LLM) used by nodes:
  - `retrieval_grader.py` — binary yes/no relevance grader (`GradeDocuments` pydantic model,
    `ChatGoogleGenerativeAI.with_structured_output`)
  - `generation.py`, `hallucination_grader.py`, `router.py` — referenced by `tests/test_chain.py` but
    **not yet created**; these will be the answer-generation chain, hallucination/groundedness
    grader, and vectorstore-vs-websearch router respectively
- `graph.py` — intended to wire the nodes above into a `StateGraph` with conditional edges
  (retrieve → grade_documents → generate, with a websearch detour). Currently empty.
- `tools/` — currently empty

### Import path gotcha

`pyproject.toml` has no `[build-system]`, and `uv.lock` marks `corrective-rag` as a virtual project
(`source = { virtual = "." }`) — the `src/graph` package is **not pip-installed**, so `import graph...`
only works if `src/` is on `sys.path`.

Two import conventions coexist and require different `sys.path` setups:

- `src/graph/node/*.py` use `from graph.xxx import ...` → needs **`src/`** on `sys.path`.
- `tests/test_chain.py` uses `from src.graph.xxx import ...` and `from ingestion import retriever`
  → needs the **project root** on `sys.path` (pytest adds this automatically because `tests/__init__.py`
  exists and the project root has none).

Running a file under `src/graph/` directly (`python src/graph/node/retrieve.py`) fails with
`ModuleNotFoundError: No module named 'graph'` because direct execution only puts the script's own
directory on `sys.path`. To run such a file manually from the project root:

```powershell
$env:PYTHONPATH = "src"
python -m graph.node.retrieve
```
