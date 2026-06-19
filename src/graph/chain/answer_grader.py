from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer address the question, 'yes' or 'no")


structure_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer address / resolves a question. Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "user question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structure_llm_grader
