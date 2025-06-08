from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import question_router
from graph.constants import GENERATE, WEBSEARCH
from graph.state import GraphState

def route_question(state: GraphState) -> str:
    res = question_router.invoke(
        {"question": state["question"]}
    )
    return res.datasource

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