import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "rag-qdrant")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embedding = OpenAIEmbeddings(model="text-embedding-3-small")


def _collection_has_data(client: QdrantClient, collection_name: str) -> bool:
    if not client.collection_exists(collection_name):
        return False
    return client.count(collection_name=collection_name, exact=True).count > 0


def _ingest_and_build_vectorstore() -> QdrantVectorStore:
    print(
        f"---INGESTION: collection '{QDRANT_COLLECTION_NAME}' missing or empty, scraping + embedding---"
    )
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    return QdrantVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION_NAME,
    )


def _attach_to_existing_vectorstore() -> QdrantVectorStore:
    print(
        f"---INGESTION: collection '{QDRANT_COLLECTION_NAME}' already populated, skipping scrape---"
    )
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding,
        collection_name=QDRANT_COLLECTION_NAME,
        url=QDRANT_URL,
    )


_client = QdrantClient(url=QDRANT_URL)

if _collection_has_data(_client, QDRANT_COLLECTION_NAME):
    vectorstore = _attach_to_existing_vectorstore()
else:
    vectorstore = _ingest_and_build_vectorstore()

retriever = vectorstore.as_retriever()
