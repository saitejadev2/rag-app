from app.retrieval.reranker import Reranker


def main():

    reranker = Reranker()

    query = "What does FastAPI provide?"

    documents = [
        "FastAPI is a modern Python web framework for building APIs.",
        "My resume includes experience with Python, React and FastAPI.",
        "FastAPI provides automatic interactive API documentation.",
        "The project uses a vector database to store document embeddings.",
    ]

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3
    )

    print("\nReranked results:\n")

    for document, score in results:
        print(f"Score: {score:.4f}")
        print(f"Document: {document}")
        print()


if __name__ == "__main__":
    main()