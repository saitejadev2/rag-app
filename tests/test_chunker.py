from app.ingestion.chunker import chunk_documents
from app.ingestion.models import Document


def test_chunk_documents_preserves_text_and_metadata():
    document = Document(
        text="A" * 1200,
        metadata={"source": "test.txt"}
    )

    chunks = chunk_documents(
        [document],
        chunk_size=500,
        chunk_overlap=50
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.text
        assert chunk.metadata["source"] == "test.txt"
        assert "chunk_id" in chunk.metadata


def test_chunk_ids_are_sequential():
    document = Document(
        text="This is a test document. " * 100,
        metadata={"source": "test.txt"}
    )

    chunks = chunk_documents(
        [document],
        chunk_size=100,
        chunk_overlap=20
    )

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    assert chunk_ids == list(range(len(chunks)))