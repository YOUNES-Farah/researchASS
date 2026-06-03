"""
ChromaDB retriever avec :
- Embeddings + DB chargés au démarrage (pas au premier appel)
- Reranker cross-encoder pour trier les docs par pertinence réelle
- Score de pertinence moyen exposé pour le fallback vectorstore→web
"""
from __future__ import annotations

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

CHROMA_PATH = "./data/chroma"

#  Embeddings (chargés une seule fois au démarrage) 
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

#  ChromaDB (chargé une seule fois) 
_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=_embeddings,
)

#  Reranker cross-encoder (léger, CPU, ~50ms) 
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#  Retriever : on récupère k=4 puis on rerank → garde top 2 
_retriever = _db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

#  Seuil de pertinence pour le fallback web (0.0 = tous les docs sont pertinents, -1.0 = aucun doc n'est pertinent)
RELEVANCE_THRESHOLD = 0.0   # score cross-encoder : négatif = non pertinent


def get_retriever():
    return _retriever


def rerank(query: str, docs: list[Document]) -> tuple[list[Document], float]:
    """
    Reranke les docs par pertinence (cross-encoder).
    Retourne (docs_triés[:2], score_moyen).
    """
    if not docs:
        return [], -999.0

    pairs = [(query, d.page_content[:512]) for d in docs]
    scores = _reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs   = [d for _, d in ranked[:2]]
    avg_score  = float(sum(s for s, _ in ranked[:2]) / max(len(ranked[:2]), 1))

    print(f"   Reranker scores: {[round(float(s),2) for s,_ in ranked]}")
    return top_docs, avg_score


def add_documents_to_db(texts: list[str], metadatas: list[dict] = None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(texts, metadatas or [{}] * len(texts))
    ]
    chunks = splitter.split_documents(docs)
    Chroma.from_documents(chunks, _embeddings, persist_directory=CHROMA_PATH)
    print(f" {len(chunks)} chunks ajoutés à ChromaDB")