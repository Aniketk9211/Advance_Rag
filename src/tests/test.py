from ingestion import retriever
from graph.chain.retrieval_grader import retrieval_grader
from graph.chain.retrieval_grader import GradeDocuments, retrieval_grader


question = "agent memory"
docs = retriever.invoke(question)
print(docs)
doc_txt = docs[0].page_content

res: GradeDocuments = retrieval_grader.invoke(
    {"question": question, "document": doc_txt})
print(res)
