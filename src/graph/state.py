from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph

    Attribuite:
        question: question
        generation: LLM generation
        web_search: Whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]  # documets retrive from the vertorstore
