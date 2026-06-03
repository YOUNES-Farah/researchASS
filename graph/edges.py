from graph.state import GraphState
from tools.retriever import RELEVANCE_THRESHOLD

# inatial routage vectorstore ou web search
def route_question(state: GraphState) -> str:
    return state.get("source", "vectorstore")

# if chromaDB ne trouve rien de pertinent → fallback web_search
def route_after_retrieval(state: GraphState) -> str:
    score = state.get("doc_relevance_score")
    if score is None or score < RELEVANCE_THRESHOLD:
        print(f"   ⚠️ Score pertinence faible ({score:.2f}) → fallback web_search")
        return "web_search"
    return "generator"