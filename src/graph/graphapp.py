from dotenv import load_dotenv

from graph.const import RETRIEVE, WEBSEARCH, GENERATE, GRADE_DOCUMENTS
from graph.state import GraphState
from graph.node import retrieve, web_search, generate, grade_documents

from langgraph.graph import END, StateGraph, END
load_dotenv()

def decide_to_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENT")

    if(state["web_search"]):
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVENT TO QUESTION, INCLUDE WEB SEARCH---")
        return WEBSEARCH
    else:
        print("--DECISION: GENERATE--")
        return GENERATE



graph = StateGraph(GraphState)

graph.add_node(RETRIEVE, retrieve)
graph.add_node(WEBSEARCH, web_search)
graph.add_node(GENERATE, generate)
graph.add_node(GRADE_DOCUMENTS, grade_documents)

graph.set_entry_point(RETRIEVE)
graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
graph.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {
    WEBSEARCH: WEBSEARCH, 
    GENERATE: GENERATE

})
graph.add_edge(WEBSEARCH, GENERATE)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")