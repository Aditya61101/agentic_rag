from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import END, StateGraph

from graph.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from conditonal_funcs import route_question, check_for_web_search, grade_generation_grounded_in_documents_and_question

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(route_question, {
    "vectorstore": RETRIEVE,
    "websearch": WEBSEARCH
})

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
