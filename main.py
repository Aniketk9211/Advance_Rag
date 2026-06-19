from dotenv import load_dotenv

load_dotenv()

from graph.graphapp import app

if __name__ == "__main__":
    print("HELLO ADAPTIVE RAG")
    print(app.invoke(input={"question": "how to make pizza without microwave."}))
