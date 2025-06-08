from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever


def retrieve(state:GraphState) -> Dict[str, Any]:
    """
        takes the question that the user has asked and retrieves relevant documents
        from the vector store and updates the state document to hold the relevant documents.
        
        Args: state (dict): The current graph state

        Returns:
            state (dict): Updated state with retrieved documents and the original question
    """
    question = state["question"]
    documents = retriever.invoke(question)

    return {'documents': documents, 'question': question}