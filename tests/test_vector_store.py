import numpy as np

from app.retrieval.vector_store import VectorStore


def test_conversation_isolation(tmp_path):
    vector_store = VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    vector_store.add(
        chunks=["Document belonging to conversation A"],
        embeddings=np.array([[1.0, 0.0, 0.0]]),
        metadatas=[
            {
                "source": "a.txt",
                "chunk_id": 0,
                "conversation_id": "conversation-A",
            }
        ],
    )

    vector_store.add(
        chunks=["Document belonging to conversation B"],
        embeddings=np.array([[0.0, 1.0, 0.0]]),
        metadatas=[
            {
                "source": "b.txt",
                "chunk_id": 0,
                "conversation_id": "conversation-B",
            }
        ],
    )

    results_a = vector_store.search(
        query_embedding=np.array([1.0, 0.0, 0.0]),
        k=5,
        conversation_id="conversation-A",
    )

    results_b = vector_store.search(
        query_embedding=np.array([0.0, 1.0, 0.0]),
        k=5,
        conversation_id="conversation-B",
    )

    assert results_a["documents"][0] == [
        "Document belonging to conversation A"
    ]

    assert results_b["documents"][0] == [
        "Document belonging to conversation B"
    ]


def test_delete_by_source_is_conversation_scoped(tmp_path):
    vector_store = VectorStore(
        persist_directory=str(tmp_path / "chroma")
    )

    vector_store.add(
        chunks=["A's resume"],
        embeddings=np.array([[1.0, 0.0, 0.0]]),
        metadatas=[
            {
                "source": "resume.pdf",
                "chunk_id": 0,
                "conversation_id": "conversation-A",
            }
        ],
    )

    vector_store.add(
        chunks=["B's resume"],
        embeddings=np.array([[0.0, 1.0, 0.0]]),
        metadatas=[
            {
                "source": "resume.pdf",
                "chunk_id": 0,
                "conversation_id": "conversation-B",
            }
        ],
    )

    vector_store.delete_by_source(
        source="resume.pdf",
        conversation_id="conversation-A",
    )

    results_a = vector_store.search(
        query_embedding=np.array([1.0, 0.0, 0.0]),
        k=5,
        conversation_id="conversation-A",
    )

    results_b = vector_store.search(
        query_embedding=np.array([0.0, 1.0, 0.0]),
        k=5,
        conversation_id="conversation-B",
    )

    assert results_a["documents"][0] == []

    assert results_b["documents"][0] == [
        "B's resume"
    ]