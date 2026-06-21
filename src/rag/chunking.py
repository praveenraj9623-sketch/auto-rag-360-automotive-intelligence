from __future__ import annotations

from typing import Iterable


def chunk_text(text: str, chunk_size: int = 140, overlap: int = 30) -> list[str]:
    words = [word for word in text.split() if word.strip()]
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than or equal to 0 and less than chunk_size")

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def chunk_document(document: dict, chunk_size: int = 140, overlap: int = 30) -> list[dict]:
    chunks = []
    for index, chunk in enumerate(chunk_text(document.get("text", ""), chunk_size, overlap), start=1):
        chunk_id = f"{document.get('id', 'document')}::chunk-{index}"
        metadata = {
            "document_id": document.get("id", "document"),
            "source": document.get("source", "unknown"),
            "document_type": document.get("document_type", "knowledge_base"),
            "sensitivity_level": document.get("sensitivity_level", "internal_engineering"),
            "DATA_SOURCE": document.get("DATA_SOURCE", "unknown"),
        }
        chunks.append({"chunk_id": chunk_id, "text": chunk, "metadata": metadata})
    return chunks


def chunk_documents(documents: Iterable[dict], chunk_size: int = 140, overlap: int = 30) -> list[dict]:
    all_chunks = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
