from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documensts"""

    binary_score: str = Field(
        ..., description="Documents are relevant to the question, 'Yes' or 'No' "
    )


llm = llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structure_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n user question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structure_llm_grader
