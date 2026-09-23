from app.retrieval.retriever import Retriever
from app.generation.generator import Generator


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def query(
    self,
    question: str,
    k: int = 3,
    conversation_id: str | None = None,
    max_distance: float | None = None
):
        results = self.retriever.retrieve(
            question,
            k=k,
            max_distance=max_distance,
            conversation_id=conversation_id
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

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

            if page is not None:
                source_info = f"{source}, page {page}"
            else:
                source_info = source

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
                    "distance": distance
                }
                for metadata, distance in zip(
                    metadatas,
                    results["distances"][0]
                )
            ]
        }