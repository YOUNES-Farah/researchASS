from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.edges import route_question, route_after_retrieval
from graph.nodes import (
    cache_check_node,
    route_cache,
    router_node,
    retriever_node,
    web_search_node,
    generator_node,
)


def build_graph():
    g = StateGraph(GraphState)

    #  Nodes 
    g.add_node("cache_check",  cache_check_node)
    g.add_node("router",       router_node)
    g.add_node("retriever",    retriever_node)
    g.add_node("web_search",   web_search_node)
    g.add_node("generator",    generator_node)

    #  Entry point 
    g.set_entry_point("cache_check")

    #  Cache hit → fin immédiate, sinon → router 
    g.add_conditional_edges(
        "cache_check",
        route_cache,
        {"end": END, "router": "router"},
    )

    #  Routage initial : vectorstore ou web search
    g.add_conditional_edges(
        "router",
        route_question,
        {"vectorstore": "retriever", "web_search": "web_search"},
    )

    #  Après retrieval ChromaDB : reranker score trop bas → web fallback 
    g.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {"generator": "generator", "web_search": "web_search"},
    )

    #  Web search → generator (toujours) 
    g.add_edge("web_search", "generator")

    #  Generator → fin 
    g.add_edge("generator", END)

    return g.compile()


graph = build_graph()