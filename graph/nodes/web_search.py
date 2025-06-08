from typing import Any, Dict
from langchain.schema import Document
from graph.state import GraphState
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results=3)
def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Performs a web search to find relevant documents based on the question.
    
    Args:
        state (GraphState): The current graph state containing the question.
    
    Returns:
        Dict[str, Any]: Updated state with web search results and the original question.
    """

    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({'query': question})
    joined_doc = "\n".join([tavily_result['content'] for tavily_result in tavily_results])
    web_search_docs = Document(page_content=joined_doc)

    if documents is not None:
        documents.append(web_search_docs)
    else:
        documents = [web_search_docs]

    return {
        'documents': documents,
        'question': question
    }