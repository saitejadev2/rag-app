from app.retrieval.retriever import Retriever
from app.generation.generator import Generator


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator
    ):
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        question: str,
        k: int = 3,
        conversation_id: str | None = None,
        max_distance: float | None = None,
        retrieved: dict | None = None
    ):
        # Use already retrieved results if provided.
        # Otherwise perform retrieval normally.
        if retrieved is None:
            retrieved = self.retriever.retrieve(
                query=question,
                k=k,
                max_distance=max_distance,
                conversation_id=conversation_id
            )

        # Use only the top-k results for generation.
        documents = retrieved["documents"][0][:k]
        metadatas = retrieved["metadatas"][0][:k]

        distances = retrieved["distances"][0][:k]

        reranker_scores = retrieved.get(
            "reranker_scores",
            [[]]
        )[0][:k]

        # No relevant documents were retrieved
        if not documents:
            return {
                "answer": (
                    "I don't have enough information in "
                    "the documents for this conversation "
                    "to answer that question."
                ),
                "sources": []
            }

        # Build context
        context_parts = []

        for document, metadata in zip(
            documents,
            metadatas
        ):
            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get("page")
            chunk_id = metadata.get("chunk_id")

            if page is not None:
                source_info = (
                    f"{source}, page {page}, "
                    f"chunk {chunk_id}"
                )
            else:
                source_info = (
                    f"{source}, "
                    f"chunk {chunk_id}"
                )

            context_parts.append(
                f"[Source: {source_info}]\n"
                f"{document}"
            )

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = self.generator.generate(
            query=question,
            context=context
        )

        return {
            "answer": answer,
            "sources": [
                {
                    **metadata,
                    "distance": distance,
                    "reranker_score": reranker_score
                }
                for metadata, distance, reranker_score in zip(
                    metadatas,
                    distances,
                    reranker_scores
                )
            ]
        }