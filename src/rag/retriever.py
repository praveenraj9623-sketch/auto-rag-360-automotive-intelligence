from __future__ import annotations

from src.data.load_datasets import load_all_documents
from src.rag.access_control import can_access, describe_policy
from src.rag.chunking import chunk_documents
from src.rag.vector_store import LocalVectorStore


class AutomotiveRetriever:
    def __init__(self, vector_store: LocalVectorStore) -> None:
        self.vector_store = vector_store

    @classmethod
    def from_chunks(cls, chunks: list[dict], embedding_mode: str = "tfidf") -> "AutomotiveRetriever":
        store = LocalVectorStore(embedding_mode=embedding_mode).build(chunks)
        return cls(store)

    def retrieve(self, query: str, role: str, top_k: int = 5) -> dict:
        results = self.vector_store.search(query=query, top_k=top_k, role=role)
        return {
            "query": query,
            "role": role,
            "access_rule": describe_policy(role),
            "results": results,
        }

    def preview_access(self, query: str, roles: list[str], top_k: int = 8) -> list[dict]:
        unrestricted = self.vector_store.search(query=query, top_k=top_k, role=None)
        preview = []
        for chunk in unrestricted:
            role_access = {role: can_access(chunk.get("metadata", {}), role) for role in roles}
            preview.append({**chunk, "role_access": role_access})
        return preview


def build_default_retriever(embedding_mode: str = "tfidf") -> AutomotiveRetriever:
    documents = load_all_documents()
    chunks = chunk_documents(documents)
    return AutomotiveRetriever.from_chunks(chunks, embedding_mode=embedding_mode)
