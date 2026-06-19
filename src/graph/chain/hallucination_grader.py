from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeHallucination(BaseModel):
    """ Binary Score for hallucination present in generation answer."""
    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes; or 'no' ")

structure_llm_grader = llm.with_structured_output(GradeHallucination)

system = """ you are a grader assessing whether an LLm generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded or supported by set of facts."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "set of facts: \n\n {documents} \n\n LLM generation: {generation}")

])

hallucination_grader: RunnableSequence = prompt | structure_llm_grader
