from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt",dangerously_pull_public_prompt=True)


generation_chain = prompt | llm | StrOutputParser()
