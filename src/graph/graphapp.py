from dotenv import load_dotenv

from graph.const import RETRIEVE, WEBSEARCH, GENERATE, GRADE_DOCUMENTS
from graph.node import retrieve, web_search, generate, grade_documents
from graph.state import GraphState
from graph.chain import answer_grader, hallucination_grader


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

def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATION---")
    question = state['question']
    generation = state["generation"]
    documents = state["documents"]

    h_result =  hallucination_grader.invoke({"documents" : documents, "generation" : generation})

    if hallucination_grade := h_result.binary_score:
        print("--- DECISION: GENERATE IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION")
        a_result = answer_grader.invoke({"question":  question, "generation": generation}) 
        if answer_grade := a_result.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("--DECISION: GENERATION DOES NOT ADDRESSES QUESTION")
            return "not useful"
    
    print("---DECISION: GENERATE IS NOT GROUNDED IN DOCUMENTS")    
    return "not supported"


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
graph.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question,{
    "useful" : END,
    "not useful": WEBSEARCH,
    "not supported" : GENERATE
})
graph.add_edge(WEBSEARCH, GENERATE)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="self_graph.png")