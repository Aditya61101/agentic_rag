from langgraph.graph import END, StateGraph

from graph.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
# from graph.chains.router import RouteQuery, question_router

def check_for_web_search(state:GraphState) -> str:
    """
    Check if web search is needed based on the state.
    """
    if state['web_search']:
        return WEBSEARCH
    return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, check_for_web_search)
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {
    "useful": END,
    "not supported": GENERATE,
    "not useful": WEBSEARCH
})
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().print_ascii()