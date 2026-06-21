from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.access_control import can_access
from src.rag.embeddings import get_embedding_model


@dataclass
class LocalVectorStore:
    embedding_mode: str = "tfidf"
    embedding_model: object | None = None
    chunks: list[dict] = field(default_factory=list)
    vectors: object | None = None

    def build(self, chunks: list[dict]) -> "LocalVectorStore":
        self.embedding_model = self.embedding_model or get_embedding_model(self.embedding_mode)
        self.chunks = list(chunks)
        texts = [chunk["text"] for chunk in self.chunks]
        self.vectors = self.embedding_model.fit_transform(texts) if texts else None
        return self

    def search(self, query: str, top_k: int = 5, role: str | None = None) -> list[dict]:
        if not query.strip() or self.vectors is None or not self.chunks:
            return []

        allowed_indices = self._allowed_indices(role)
        if not allowed_indices:
            return []

        query_vector = self.embedding_model.transform([query])
        candidate_vectors = self.vectors[allowed_indices]
        scores = cosine_similarity(query_vector, candidate_vectors).ravel()
        if scores.size == 0:
            return []

        ranked_positions = np.argsort(scores)[::-1][:top_k]
        results = []
        for position in ranked_positions:
            score = float(scores[position])
            if score <= 0:
                continue
            chunk = dict(self.chunks[allowed_indices[position]])
            chunk["metadata"] = dict(chunk.get("metadata", {}))
            chunk["score"] = score
            results.append(chunk)
        return results

    def _allowed_indices(self, role: str | None) -> list[int]:
        if role is None:
            return list(range(len(self.chunks)))
        return [index for index, chunk in enumerate(self.chunks) if can_access(chunk.get("metadata", {}), role)]
