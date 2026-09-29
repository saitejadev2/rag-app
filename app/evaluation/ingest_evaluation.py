from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent.parent

EVALUATION_FILE = (
    BASE_DIR
    / "data"
    / "documents"
    / "evaluation"
    / "evaluation.txt"
)

CHROMA_DIR = (
    BASE_DIR
    / "data"
    / "chroma"
)


def main():

    print(
        f"Processing: "
        f"{EVALUATION_FILE.name}"
    )

    embedder = Embedder()

    vector_store = VectorStore(
        persist_directory=str(CHROMA_DIR)
    )

    conversation_id = "evaluation"

    # Remove an older evaluation copy first
    vector_store.delete_by_source(
        EVALUATION_FILE.name,
        conversation_id=conversation_id
    )

    # Load document
    documents = load_document(
        str(EVALUATION_FILE)
    )

    # Chunk document
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=50
    )

    # Add conversation scope
    for chunk in chunks:

        chunk.metadata[
            "conversation_id"
        ] = conversation_id

    # Create embeddings
    texts = [
        chunk.text
        for chunk in chunks
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(
        texts
    )

    # Store in Chroma
    vector_store.add(
        chunks=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print()
    print("----------------------------")
    print(
        f"Evaluation ingestion complete!"
    )
    print(
        f"Total chunks: {len(chunks)}"
    )
    print("----------------------------")


if __name__ == "__main__":
    main()