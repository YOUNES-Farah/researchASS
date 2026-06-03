from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    question: str
    generation: str
    documents: List[Document]
    source: str
    cache_hit: bool
    doc_relevance_score: float