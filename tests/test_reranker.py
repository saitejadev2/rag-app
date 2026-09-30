from app.retrieval.reranker import Reranker


def test_reranker_orders_relevant_document_first():
    reranker = Reranker()

    query = "What is FastAPI?"

    documents = [
        "PostgreSQL is a relational database.",
        "FastAPI is a modern Python web framework for building APIs.",
        "Redis is an in-memory data store.",
    ]

    results = reranker.rerank(
        query=query,
        documents=documents,
        top_k=3,
    )

    ranked_documents = [
        document
        for document, score in results
    ]

    assert ranked_documents[0] == (
        "FastAPI is a modern Python web framework for building APIs."
    )