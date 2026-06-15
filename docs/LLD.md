# Low-Level Design — Corrective RAG (CRAG)

Status legend: **[DONE]** implemented and roughly correct · **[BUGGY]** implemented but broken ·
**[TODO]** not implemented yet.

---

## 1. Package layout

```
ingestion.py                          [DONE]   builds/loads Chroma vectorstore, exposes `retriever`
main.py                               [TODO]   entry point — should build the graph and invoke it
src/graph/
  state.py                            [DONE]   GraphState TypedDict (shared state)
  graph.py                            [TODO]   StateGraph wiring (nodes + edges)
  chain/
    retrieval_grader.py               [DONE]   per-document relevance grader (LLM, structured output)
    generation.py                     [BUGGY]  RAG answer generation chain (bad model id)
    hallucination_grader.py           [TODO]   groundedness grader (generation vs documents)
    router.py                         [TODO]   question -> "vectorstore" | "websearch"
  node/
    retrieve.py                       [DONE]   vectorstore retrieval node
    grade_documents.py                [BUGGY]  filters docs + sets web_search flag
    web_search.py                     [DONE]   Tavily fallback search node
    generate.py                       [TODO]   final-answer generation node (wraps generation_chain)
  tools/                              [TODO]   empty, unused
```

---

## 2. Shared state — `GraphState` (`src/graph/state.py`)

```python
class GraphState(TypedDict):
    question: str          # user's question, set once at graph entry
    generation: str         # final LLM answer (set by `generate` node)
    web_search: bool        # True if grade_documents found ≥1 irrelevant doc
    documents: List[Document]  # working set of retrieved/augmented documents
```

Every node receives the **full** `GraphState` dict and returns a **partial** dict of the keys it
updates. LangGraph merges the partial dict into the running state (shallow merge — list/bool values
are replaced wholesale, not appended, except where a node explicitly reads-then-writes the list).

Note: `documents` is annotated `List[str]` in the current code but actually holds
`langchain_core.documents.Document` objects everywhere it's used — this should be corrected to
`List[Document]`.

---

## 3. Ingestion layer (`ingestion.py`)

| Item | Detail |
|---|---|
| Trigger | Module-level side effect on first `import ingestion` (or `from ingestion import retriever`) |
| Sources | 3 hardcoded Lilian Weng blog URLs, loaded via `WebBaseLoader` |
| Chunking | `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)` |
| Embeddings | `GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")` |
| Store | `Chroma(collection_name="rag-chroma", persist_directory="./.chroma")` |
| Idempotency | `ingest_documents()` only runs `vectorstore.add_documents(...)` if `vectorstore._collection.count() == 0` |
| Retry | `@retry` wraps `ingest_documents`, retries on `GoogleGenerativeAIError`, exponential backoff (2s→60s), 5 attempts |
| Export | `retriever = vectorstore.as_retriever()` — module-level singleton imported by `retrieve.py` and tests |

**Dependency note**: any module that imports `retriever` (directly or transitively) pays the cost
of this ingestion check at import time. This includes `node/retrieve.py` and `tests/test_chain.py`.

---

## 4. Chains (`src/graph/chain/`)

### 4.1 `retrieval_grader.py` — [DONE]

```python
class GradeDocuments(BaseModel):
    binary_score: str   # "yes" | "no"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
structure_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structure_llm_grader
```

- **Input**: `{"question": str, "document": str}` (a single document's `page_content`)
- **Output**: `GradeDocuments(binary_score="yes"|"no")`
- **Used by**: `node/grade_documents.py`, once per retrieved document.

### 4.2 `generation.py` — [BUGGY]

```python
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")   # BUG: not a real model id
prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
generation_chain = prompt | llm | StrOutputParser()
```

- **Input**: `{"context": List[Document], "question": str}`
- **Output**: `str` (final answer)
- **Fix required**: replace `"gemini-3.5-flash"` with a valid Gemini chat model (e.g.
  `"gemini-1.5-flash"` or `"gemini-2.0-flash"`), consistent with `retrieval_grader.py`.
- **Used by**: `node/generate.py` (not yet created).

### 4.3 `hallucination_grader.py` — [TODO]

Mirrors `retrieval_grader.py`'s structure:

```python
class GradeHallucinations(BaseModel):
    binary_score: bool   # True = generation is grounded in documents

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = (
    "You are a grader assessing whether an LLM generation is grounded in / "
    "supported by a set of retrieved facts. Give a binary score: True means "
    "the generation is grounded in the facts."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

hallucination_grader = prompt | structured_llm_grader
```

- **Input**: `{"documents": List[Document], "generation": str}`
- **Output**: `GradeHallucinations(binary_score: bool)`
- **Used by**: a (currently non-existent) `grade_generation_v_documents` conditional edge after
  `generate`, to decide whether to re-try generation, fall back to web search, or finish.
  Referenced by commented-out tests in `src/tests/test_chain.py`.

### 4.4 `router.py` — [TODO]

```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
structured_llm_router = llm.with_structured_output(RouteQuery)

system = (
    "You are an expert at routing a user question to a vectorstore or web search. "
    "The vectorstore contains documents related to agents, prompt engineering, and "
    "adversarial attacks on LLMs. Use the vectorstore for questions on these topics, "
    "otherwise use web search."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

question_router = prompt | structured_llm_router
```

- **Input**: `{"question": str}`
- **Output**: `RouteQuery(datasource: "vectorstore"|"websearch")`
- **Used by**: the graph's **entry-point conditional edge** — decides whether to start at
  `retrieve` or skip straight to `web_search`. Referenced by commented-out tests
  (`test_router_to_vectorstore`, `test_router_to_websearch`).

---

## 5. Nodes (`src/graph/node/`)

All node functions share the signature `def node_fn(state: GraphState) -> Dict[str, Any]`.

### 5.1 `retrieve.py` — [DONE]

```python
def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
```

- **Reads**: `question`
- **Writes**: `documents`
- **Side effects**: none beyond the vectorstore similarity search.

### 5.2 `grade_documents.py` — [BUGGY]

Intended behavior (per docstring): grade every retrieved document; keep only the relevant ones;
set `web_search=True` if **any** document is irrelevant.

Current bugs to fix:

1. `def grade_documents(state: GraphState) -> Dict(str, Any):` → must be `Dict[str, Any]`
   (`Dict(...)` is a call, not a subscript — this is a syntax/runtime error).
2. `from numpy import False_` — unused/unnecessary import; remove. Use the builtin `False`.
3. The `return` statement is nested **inside** the `for` loop, so the function returns after the
   first document and never grades the rest. It must be **outside/after** the loop.
4. The print statements have swapped/confusing labels (says "DOCUMENT NOT RELEVANT" on the
   "yes"/keep branch). Cosmetic, but should be corrected for debuggability.

Corrected shape:

```python
def grade_documents(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for document in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        if score.binary_score.lower() == "yes":
            filtered_docs.append(document)
        else:
            web_search = True

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }
```

- **Reads**: `question`, `documents`
- **Writes**: `documents` (filtered), `web_search`

### 5.3 `web_search.py` — [DONE]

```python
web_serch_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_serch_tool.invoke({"query": question})
    joined = "\n".join(r["content"] for r in tavily_results["results"])
    web_results = Document(page_content=joined)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}
```

- **Reads**: `question`, `documents`
- **Writes**: `documents` (appended with one synthetic `Document` of joined search results)
- **External call**: Tavily Search API (`TAVILY_API_KEY`)
- Minor naming nit: `web_serch_tool` → `web_search_tool` (no functional impact).

### 5.4 `generate.py` — [TODO]

```python
from graph.chain.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke(
        {"context": documents, "question": question}
    )
    return {"documents": documents, "question": question, "generation": generation}
```

- **Reads**: `question`, `documents`
- **Writes**: `generation`
- **External call**: Gemini chat completion (`GOOGLE_API_KEY`)

---

## 6. Graph wiring (`src/graph/graph.py`) — [TODO]

### 6.1 Node registration

| Node key | Function |
|---|---|
| `"retrieve"` | `retrieve` |
| `"grade_documents"` | `grade_documents` |
| `"websearch"` | `web_search` |
| `"generate"` | `generate` |

### 6.2 Edges

```
ENTRY ──(decide_entry_route)──┬─► "websearch" ──► "generate" ──► END
                               └─► "retrieve" ──► "grade_documents" ──(decide_to_generate)──┬─► "websearch" ──► "generate" ──► END
                                                                                              └─► "generate" ──► END
```

- **Entry conditional edge** — `route_question(state) -> Literal["websearch", "retrieve"]`
  - Calls `question_router.invoke({"question": state["question"]})`.
  - `datasource == "websearch"` → go to `"websearch"`.
  - `datasource == "vectorstore"` → go to `"retrieve"`.
  - *(If `router.py` is not built yet, the entry point can be hardcoded to `"retrieve"` as an interim
    simplification.)*

- **Conditional edge after `grade_documents`** — `decide_to_generate(state) -> Literal["websearch", "generate"]`
  - `state["web_search"] is True` → `"websearch"` (one or more retrieved docs were irrelevant —
    augment with live search before answering).
  - else → `"generate"`.

- **`"websearch"` → `"generate"`**: normal edge (always).
- **`"generate"` → `END`**: normal edge for v1. A future iteration could add the
  `grade_generation_v_documents` conditional edge using `hallucination_grader` to loop back to
  `"generate"` or `"websearch"` on a failed groundedness check — out of scope for this LLD pass but
  noted as the natural extension point.

### 6.3 Skeleton

```python
from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.node.retrieve import retrieve
from graph.node.grade_documents import grade_documents
from graph.node.web_search import web_search
from graph.node.generate import generate
# from graph.chain.router import question_router  # once implemented


def decide_to_generate(state: GraphState) -> str:
    return "websearch" if state["web_search"] else "generate"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")  # replace with conditional router entry once available
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"websearch": "websearch", "generate": "generate"},
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

---

## 7. Entry point (`main.py`) — [TODO]

```python
from dotenv import load_dotenv
from graph.graph import app

load_dotenv()


def main():
    result = app.invoke({"question": "What is agent memory?"})
    print(result["generation"])


if __name__ == "__main__":
    main()
```

Run via:

```powershell
$env:PYTHONPATH = "src"
uv run python main.py
```

(per the import-path conventions documented in `CLAUDE.md`).

---

## 8. External dependencies / config

| Service | Env var | Used by |
|---|---|---|
| Google Gemini (chat + embeddings) | `GOOGLE_API_KEY` | `ingestion.py`, `retrieval_grader.py`, `generation.py`, `hallucination_grader.py` (todo), `router.py` (todo) |
| Tavily Search | `TAVILY_API_KEY` | `web_search.py` |
| LangSmith | `LANGSMITH_*` | `generation.py` (`client.pull_prompt`), tracing |
| Groq | `GROQ_API_KEY` | declared in `.env` per CLAUDE.md, not currently referenced by any module |
| Pinecone | `PINECONE_API_KEY` | declared in `.env` per CLAUDE.md, not currently referenced (Chroma is the actual store) |

---

## 9. Sequence flows

### 9.1 Happy path — all retrieved docs relevant

```
main.invoke(question)
 └─ retrieve            : documents = retriever.invoke(question)
 └─ grade_documents      : all docs score "yes" -> web_search=False, documents unchanged
 └─ generate             : generation = generation_chain.invoke({context: documents, question})
 └─ END (state.generation returned)
```

### 9.2 Corrective path — at least one doc irrelevant

```
main.invoke(question)
 └─ retrieve            : documents = retriever.invoke(question)
 └─ grade_documents      : ≥1 doc scores "no" -> web_search=True, documents = filtered relevant docs
 └─ websearch            : Tavily search -> append synthetic Document to documents
 └─ generate             : generation = generation_chain.invoke({context: documents, question})
 └─ END
```

### 9.3 Router-direct path (once `router.py` exists)

```
main.invoke(question)
 └─ route_question       : question_router -> "websearch" (off-topic question)
 └─ websearch            : Tavily search -> documents = [web_results]
 └─ generate             : generation = generation_chain.invoke({context: documents, question})
 └─ END
```

---

## 10. Summary of fixes needed to make this LLD's "DONE/BUGGY" parts actually runnable

1. `src/graph/node/grade_documents.py` — fix `Dict(str, Any)` → `Dict[str, Any]`, remove the
   `numpy` import, move `return` outside the `for` loop, fix print labels.
2. `src/graph/chain/generation.py` — fix model id `"gemini-3.5-flash"` → a valid Gemini model.
3. `src/graph/state.py` — `documents: List[str]` → `documents: List[Document]` (type-accuracy only).
4. Create `src/graph/node/generate.py`, `src/graph/chain/router.py`,
   `src/graph/chain/hallucination_grader.py`, and `src/graph/graph.py` per §4–6 above.
5. Wire `main.py` to build and invoke the compiled graph (§7).
