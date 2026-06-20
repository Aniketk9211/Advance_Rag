# Corrective RAG

A [LangGraph](https://langchain-ai.github.io/langgraph/) implementation of **Self-/Corrective RAG**:
retrieve from a vector store, grade each retrieved document for relevance, fall back to a Tavily web
search when the documents aren't good enough, generate an answer, and grade that generation for
hallucination and for whether it actually answers the question — looping back to regenerate or
web-search again if not.

## How it works

```
                question
                   │
                   ▼
           route_question ──────────────► websearch
                   │                          │
              vectorsearch                    │
                   │                          │
                   ▼                          │
               retrieve                       │
                   │                          │
                   ▼                          │
           grade_documents                    │
             │            │                   │
       all relevant   some irrelevant ─────────┘
             │
             ▼
           generate ◄────────────────────┐
             │                           │
             ▼                           │
   grade_generation_grounded_in          │
   _documents_and_question               │
       │        │         │              │
    useful  not useful  not supported ────┘
       │        │
      END   websearch
```

1. **Route** — an LLM decides whether the question belongs in the vector store
   (`src/graph/chain/question_router.py`) or should go straight to web search.
2. **Retrieve** — pull documents from the Qdrant vector store.
3. **Grade documents** — an LLM grades each document for relevance; if any are irrelevant, flag the
   state to fall back to web search.
4. **Web search** — Tavily search results are appended to the document set when the vector store
   doesn't have enough relevant context.
5. **Generate** — produce an answer from the current documents.
6. **Grade generation** — check the answer is grounded in the documents (no hallucination) and that
   it actually addresses the question. Ungrounded answers route back to `generate`; answers that
   don't address the question route to `websearch` for another pass.

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph orchestration
- [LangChain](https://github.com/langchain-ai/langchain) + `langchain-openai` — LLM chains (`gpt-4o-mini`) and embeddings
- [Qdrant](https://qdrant.tech/) — vector store (runs as its own container via Docker Compose)
- [Tavily](https://tavily.com/) (`langchain-tavily`) — web search fallback
- [LangSmith](https://smith.langchain.com/) — tracing and prompt hub
- [uv](https://docs.astral.sh/uv/) — dependency management
- [Docker](https://www.docker.com/) / Docker Compose — containerized app + vector store
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) — HTTP API serving the graph

## Project structure

```
main.py                        # one-shot CLI entry point: invokes the compiled graph once
api.py                          # FastAPI app: serves the compiled graph over HTTP
src/
  ingestion.py                  # loads/splits source docs, exposes `retriever`
  graph/
    state.py                   # GraphState TypedDict shared across nodes
    const.py                   # node name / route constants
    graphapp.py                # builds and compiles the StateGraph
    chain/                     # LCEL chains (prompt | structured-output LLM)
      question_router.py
      retrieval_grader.py
      generation.py
      hallucination_grader.py
      answer_grader.py
    node/                      # LangGraph node functions
      retrieve.py
      grade_documents.py
      web_search.py
      generate.py
  tests/
    test_chain.py
```

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/), or Docker + Docker Compose.

Create a `.env` file in the project root (copy `.env.example` and fill in real values):

| Variable          | Required | Used for                                  |
|-------------------|----------|--------------------------------------------|
| `OPENAI_API_KEY`  | yes      | all LLM calls and embeddings               |
| `TAVILY_API_KEY`  | yes      | web search fallback node                   |
| `LANGSMITH_API_KEY` | yes    | tracing, and pulling the generation prompt from the LangSmith hub |
| `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT` | optional | tracing configuration |
| `QDRANT_URL` | optional | Qdrant connection URL (defaults to `http://qdrant:6333`, the Docker Compose service address) |
| `QDRANT_COLLECTION_NAME` | optional | Qdrant collection name (defaults to `rag-qdrant`) |

The vector store retriever (`src/ingestion.py`) connects to Qdrant and checks whether its collection
already has data. If the collection is missing or empty, it automatically scrapes and embeds the
source documents on startup; otherwise it attaches to the existing collection with no re-ingestion.

### Option A: Docker Compose (recommended)

```powershell
docker compose up --build
```

This starts a `qdrant` container and an `app` container. On first run, `app` automatically ingests the
three source blog posts into Qdrant (since the collection starts empty); on subsequent runs it detects
the existing collection and skips re-ingestion. Qdrant data persists across runs in a named Docker
volume (`qdrant_storage`).

### Option B: Local with uv

```powershell
uv sync
```

Requires a reachable Qdrant instance — either run one standalone:

```powershell
docker run -p 6333:6333 qdrant/qdrant:latest
```

and set `QDRANT_URL=http://localhost:6333` in `.env`, or start just the Qdrant service from Compose:

```powershell
docker compose up qdrant
```

## Usage

`docker compose up` runs the FastAPI server (`api.py`), which stays up and can answer many questions
without restarting or re-ingesting:

```powershell
docker compose up --build -d
```

```powershell
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{\"question\": \"What is prompt engineering?\"}'
```

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs` once the server is up.

For a quick one-shot CLI sanity check instead (hardcoded question, prints once and exits):

```powershell
uv run python main.py
# or, via Docker Compose:
docker compose run --rm app python main.py
```

## API

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/health` | GET | — | `{"status": "ok"}` |
| `/ask` | POST | `{"question": string}` | `{"question": string, "answer": string, "sources": string[]}` |

`sources` is the page content of each document the graph used to ground the answer (from the vector
store and/or web search). CORS is open to all origins for now (ahead of the planned React frontend);
tighten this before any real deployment.

## Testing

```powershell
uv run pytest
uv run pytest src/tests/test_chain.py::test_question_router_to_websearch  # run a single test
```

Or inside the container (requires `qdrant` to be running, e.g. via `docker compose up -d qdrant` first):

```powershell
docker compose run --rm app uv run pytest
```

## Linting

```powershell
uv run ruff check .
```
