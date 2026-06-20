from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.graphapp import app as rag_graph

app = FastAPI(title="Corrective RAG API")

# Permissive for now -- the planned React frontend will hit this from a different origin;
# tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = rag_graph.invoke(input={"question": request.question})
    return AskResponse(
        question=result["question"],
        answer=result["generation"],
        sources=[doc.page_content for doc in result["documents"]],
    )
