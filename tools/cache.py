"""
Cache sémantique : stocke les paires (question, réponse).
À la prochaine question similaire (cosine > seuil), on retourne
la réponse en cache sans appeler Groq.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.92   # ← ajuste selon tes besoins

@dataclass
class CacheEntry:
    question: str
    answer: str
    source: str
    embedding: np.ndarray
    ts: float = field(default_factory=time.time)


class SemanticCache:
    def __init__(self, threshold: float = _SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._model = SentenceTransformer(_MODEL_NAME)
        self._entries: list[CacheEntry] = []

    def _embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def get(self, question: str) -> Optional[CacheEntry]:
        if not self._entries:
            return None
        q_emb = self._embed(question)
        scores = np.array([
            float(np.dot(q_emb, e.embedding))
            for e in self._entries
        ])
        best_idx = int(np.argmax(scores))
        if scores[best_idx] >= self.threshold:
            print(f"✅ Cache hit (score={scores[best_idx]:.3f})")
            return self._entries[best_idx]
        return None

    def set(self, question: str, answer: str, source: str) -> None:
        emb = self._embed(question)
        self._entries.append(CacheEntry(question, answer, source, emb))

    def clear(self) -> None:
        self._entries.clear()


# ✅ Singleton partagé dans tout le projet
semantic_cache = SemanticCache()