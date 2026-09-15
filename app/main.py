from pathlib import Path

from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from generation.generator import Generator
from rag.pipeline import RAGPipeline


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"


def main():

    # Load embedding model
    embedder = Embedder()

    # Connect to existing vector database
    vector_store = VectorStore(
        persist_directory=str(CHROMA_DIR)
    )

    # Create retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder
    )

    # Create generator
    generator = Generator()

    # Create RAG pipeline
    rag = RAGPipeline(
        retriever=retriever,
        generator=generator
    )

    # Ask question
    question = input("\nAsk a question: ")

    result = rag.query(
        question,
        k=3
    )

    # Print answer
    print("\n================ ANSWER ================\n")
    print(result["answer"])

    # Print sources
    print("\n================ SOURCES ================\n")

    for source in result["sources"]:
        filename = source.get("source", "Unknown")
        page = source.get("page")

        if page:
            print(f"- {filename}, page {page}")
        else:
            print(f"- {filename}")


if __name__ == "__main__":
    main()