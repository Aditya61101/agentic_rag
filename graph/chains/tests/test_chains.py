from pprint import pprint

from dotenv import load_dotenv
load_dotenv()

from ingestion import retriever
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
# from graph.chains.hallucination_grader import (GradeHallucinations,
#                                                hallucination_grader)
# from graph.chains.router import RouteQuery, question_router


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)