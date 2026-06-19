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
2. **Retrieve** — pull documents from the local Chroma vector store.
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
- [Chroma](https://github.com/chroma-core/chroma) — local vector store
- [Tavily](https://tavily.com/) (`langchain-tavily`) — web search fallback
- [LangSmith](https://smith.langchain.com/) — tracing and prompt hub
- [uv](https://docs.astral.sh/uv/) — dependency management

## Project structure

```
main.py                        # entry point: invokes the compiled graph
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

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
```

Create a `.env` file in the project root with:

| Variable          | Required | Used for                                  |
|-------------------|----------|--------------------------------------------|
| `OPENAI_API_KEY`  | yes      | all LLM calls and embeddings               |
| `TAVILY_API_KEY`  | yes      | web search fallback node                   |
| `LANGSMITH_API_KEY` | yes    | tracing, and pulling the generation prompt from the LangSmith hub |
| `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT` | optional | tracing configuration |

The vector store retriever opens an existing Chroma collection at `./.chroma`. If that directory
doesn't exist yet, uncomment the ingestion block in `src/ingestion.py` and run it once to scrape and
embed the source documents before querying the vector store.

## Usage

```powershell
uv run python main.py
```

## Testing

```powershell
uv run pytest
uv run pytest src/tests/test_chain.py::test_question_router_to_websearch  # run a single test
```

## Linting

```powershell
uv run ruff check .
```
