from src.rag.chunking import chunk_documents
from src.rag.retriever import AutomotiveRetriever


def test_retriever_returns_non_empty_results_for_known_query() -> None:
    documents = [
        {
            "id": "recall-1",
            "text": "Brake actuator recall remedy replaces the hydraulic control unit and recalibrates brakes.",
            "source": "fixture",
            "document_type": "recall_notice",
            "sensitivity_level": "public",
        },
        {
            "id": "notice-1",
            "text": "Supplier notice requests fixture checks for brake hose brackets.",
            "source": "fixture",
            "document_type": "supplier_notice",
            "sensitivity_level": "supplier_visible",
        },
    ]
    chunks = chunk_documents(documents, chunk_size=30, overlap=5)
    retriever = AutomotiveRetriever.from_chunks(chunks)

    retrieval = retriever.retrieve("brake actuator remedy", role="Supplier", top_k=3)

    assert retrieval["results"]
    assert "Brake actuator recall remedy" in retrieval["results"][0]["text"]
