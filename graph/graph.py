from langgraph.graph import END, StateGraph

from graph.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()
# from graph.chains.answer_grader import answer_grader
# from graph.chains.hallucination_grader import hallucination_grader
# from graph.chains.router import RouteQuery, question_router
def check_condition(state:GraphState) -> str:
    """
    Check if web search is needed based on the state.
    """
    if state['web_search']:
        return WEBSEARCH
    return GENERATE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, check_condition, {
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
})
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

workflow.set_entry_point(RETRIEVE)

app = workflow.compile()
app.get_graph().print_ascii()