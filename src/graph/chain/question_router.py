from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from typing import Literal
from pydantic import BaseModel, Field

class RouterQuery(BaseModel):
    route: Literal["vectorsearch", "websearch"] = Field(
        ..., 
        description = "Given a user's question to route it to web search or a vectorstore.")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_structured_output = llm.with_structured_output(RouterQuery)

system = """you are an expert at routing a user queston to a vertorstore or websearch. The vecotrstore contains documents releted to agent, prompt engineering, advarsarial attacks. use the vectorstore for the question on these topic. for all else, use web-search."""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

question_router = router_prompt | llm_structured_output