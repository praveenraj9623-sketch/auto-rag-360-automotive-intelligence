from __future__ import annotations

import os
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingModel:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)

    def fit_transform(self, texts: Iterable[str]):
        return self.vectorizer.fit_transform(list(texts))

    def transform(self, texts: Iterable[str]):
        return self.vectorizer.transform(list(texts))


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers mode is optional. Install sentence-transformers separately "
                "and set AUTORAG_EMBEDDING_MODE=sentence-transformers to use it."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: Iterable[str]):
        return self.model.encode(list(texts), normalize_embeddings=True)

    def transform(self, texts: Iterable[str]):
        return self.model.encode(list(texts), normalize_embeddings=True)


def get_embedding_model(mode: str | None = None):
    selected_mode = (mode or os.getenv("AUTORAG_EMBEDDING_MODE", "tfidf")).strip().lower()
    if selected_mode in {"tfidf", "tf-idf"}:
        return TfidfEmbeddingModel()
    if selected_mode in {"sentence-transformers", "sentence_transformers"}:
        model_name = os.getenv("AUTORAG_SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddingModel(model_name=model_name)
    raise ValueError(f"Unsupported embedding mode: {selected_mode}")
