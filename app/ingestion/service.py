from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


class IngestionService:

    def __init__(
        self,
        documents_dir: Path,
        chroma_dir: Path
    ):
        self.documents_dir = documents_dir

        self.embedder = Embedder()

        self.vector_store = VectorStore(
            persist_directory=str(chroma_dir)
        )

    def ingest_file(
    self, file_path: Path, conversation_id: str | None = None):

        print(f"Processing: {file_path.name}")

        # Remove old version of this document
        self.vector_store.delete_by_source(
            file_path.name,
            conversation_id=conversation_id
        )

        # Load document
        documents = load_document(
            str(file_path)
        )

        # Create chunks
        chunks = chunk_documents(
            documents,
            chunk_size=500,
            chunk_overlap=50
        )
        for chunk in chunks:
            if conversation_id is not None:
                chunk.metadata["conversation_id"] = conversation_id

        if not chunks:
            raise ValueError(
                "No text could be extracted from the document."
            )

        # Extract text and metadata
        texts = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            chunk.metadata
            for chunk in chunks
        ]

        # Create embeddings
        embeddings = self.embedder.embed_documents(
            texts
        )

        # Store in vector database
        self.vector_store.add(
            chunks=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return {
            "filename": file_path.name,
            "chunks": len(chunks)
        }