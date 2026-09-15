from pathlib import Path

from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma"


def ingest_documents():
    embedder = Embedder()
    vector_store = VectorStore(
        persist_directory=str(CHROMA_DIR)
    )

    files = list(DOCUMENTS_DIR.glob("*.txt"))
    files += list(DOCUMENTS_DIR.glob("*.pdf"))

    if not files:
        print("No documents found.")
        return

    total_chunks = 0

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        vector_store.delete_by_source(file_path.name)

        # 1. Load the document
        documents = load_document(str(file_path))

        print(f"Loaded {len(documents)} document sections.")

        # 2. Split into chunks while preserving metadata
        chunks = chunk_documents(
            documents,
            chunk_size=500,
            chunk_overlap=50
        )

        print(f"Created {len(chunks)} chunks.")

        # 3. Extract text and metadata
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # 4. Generate embeddings
        embeddings = embedder.embed_documents(texts)

        # 5. Store everything in Chroma
        vector_store.add(
            chunks=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(chunks)

    print("\n----------------------------")
    print("Ingestion complete!")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Vector store size: {vector_store.size()}")
    print("----------------------------")


if __name__ == "__main__":
    ingest_documents()