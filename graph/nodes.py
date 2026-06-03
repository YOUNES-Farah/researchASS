from __future__ import annotations
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from groq import RateLimitError

from graph.state import GraphState
from prompts import GENERATOR_PROMPT
from tools.retriever import get_retriever, rerank
from tools.web_search import web_search
from tools.cache import semantic_cache
from dotenv import load_dotenv

load_dotenv()

#  LLM : 8b pour questions simples, 70b pour complexes 
_llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
)
_llm_smart = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=512,
)

# Mots-clés = question complexe → 70b
_COMPLEX_KEYWORDS = {
    "explique", "explain", "compare", "analyse", "analyse", "pourquoi", "why",
    "comment", "how", "différence", "difference", "avantage", "inconvénient",
    "pros", "cons", "détaille", "detail",
}

def _pick_llm(question: str):
    q = question.lower()
    if any(kw in q for kw in _COMPLEX_KEYWORDS):
        print("   🧠 Modèle : llama-3.3-70b (question complexe)")
        return _llm_smart
    print("   ⚡ Modèle : llama-3.1-8b (question simple)")
    return _llm_fast


#  Mots-clés forçant une recherche web 
_WEB_KEYWORDS = {
    "aujourd'hui", "actuellement", "récemment", "dernières nouvelles",
    "2024", "2025", "2026", "prix actuel", "météo", "score", "live",
    "latest", "recent", "news", "current", "today", "now",
}


#  Retry Groq (rate limit) 
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _invoke_llm(chain, payload: dict) -> str:
    return chain.invoke(payload)


# NODES

def cache_check_node(state: GraphState) -> GraphState:
    """Vérifie le cache sémantique AVANT tout appel réseau."""
    entry = semantic_cache.get(state["question"])
    if entry:
        return {
            "generation": entry.answer,
            "source": entry.source,
            "cache_hit": True,
            "documents": [],
            "doc_relevance_score": 1.0,
        }
    return {"cache_hit": False}


def route_cache(state: GraphState) -> str:
    # Si cache hit → on saute tout et on va directement à END.
    return "end" if state.get("cache_hit") else "router"


def router_node(state: GraphState) -> GraphState:
    # Routage par mots-clés (zéro appel LLM).
    q = state["question"].lower()
    source = "web_search" if any(kw in q for kw in _WEB_KEYWORDS) else "vectorstore"
    print(f"🔀 Router → {source}")
    return {"source": source}


def retriever_node(state: GraphState) -> GraphState:
    # Retrieval ChromaDB + reranking cross-encoder.
    print("📚 Recherche ChromaDB...")
    retriever = get_retriever()
    raw_docs = retriever.invoke(state["question"])
    print(f"   {len(raw_docs)} docs bruts")

    ranked_docs, avg_score = rerank(state["question"], raw_docs)
    print(f"   Score moyen reranker : {avg_score:.2f}")

    return {
        "documents": ranked_docs,
        "doc_relevance_score": avg_score,
    }


def web_search_node(state: GraphState) -> GraphState:
    # Recherche web via Tavily (ou DuckDuckGo en fallback).
    print("🌐 Recherche web...")
    docs = web_search(state["question"])
    return {
        "documents": docs,
        "doc_relevance_score": 1.0,   # web = toujours pertinent par définition
    }


def generator_node(state: GraphState) -> GraphState:
    # Génération avec le bon modèle + retry tenacity.
    print("💬 Génération...")

    MAX_CHARS = 150
    docs = state.get("documents", [])
    docs_text = "\n\n".join(
        f"[{i+1}] {d.page_content[:MAX_CHARS]}"
        for i, d in enumerate(docs)
    )
    if not docs_text.strip():
        docs_text = "Réponds avec tes connaissances générales."

    llm   = _pick_llm(state["question"])
    chain = GENERATOR_PROMPT | llm | StrOutputParser()

    answer = _invoke_llm(chain, {
        "question": state["question"],
        "context":  docs_text,
    })

    # Mise en cache de la réponse
    semantic_cache.set(state["question"], answer, state.get("source", "vectorstore"))

    return {"generation": answer}