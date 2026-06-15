from dotenv import load_dotenv

load_dotenv()

from graph.graphapp import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "agent memory?"}))