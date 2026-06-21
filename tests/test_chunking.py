from src.rag.chunking import chunk_documents, chunk_text


def test_chunking_produces_non_empty_chunks() -> None:
    chunks = chunk_text("brake actuator remedy " * 80, chunk_size=25, overlap=5)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_documents_preserves_metadata() -> None:
    documents = [
        {
            "id": "doc-1",
            "text": "air bag control module wiring inspection " * 20,
            "source": "fixture",
            "document_type": "service_manual",
            "sensitivity_level": "internal_engineering",
        }
    ]
    chunks = chunk_documents(documents, chunk_size=20, overlap=4)
    assert chunks
    assert chunks[0]["metadata"]["document_type"] == "service_manual"
