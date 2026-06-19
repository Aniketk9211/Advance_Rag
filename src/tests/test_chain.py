from dotenv import load_dotenv

load_dotenv()
from pprint import pprint
from graph.chain.generation import generation_chain
from graph.chain.hallucination_grader import hallucination_grader, GradeHallucination

from graph.chain.retrieval_grader import GradeDocuments, retrieval_grader

# from graph.chain.router import RouteQuery, question_router
from graph.chain import question_router
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


def test_question_router_to_websearch() -> None:
    question = "how to make pizza"
    res = question_router.invoke({"question": question})
    assert res.route == "websearch"


def test_question_router_to_vectorstore() -> None:
    question = "agent memory"
    res = question_router.invoke({"question": question})

    assert res.route == "vectorsearch"
