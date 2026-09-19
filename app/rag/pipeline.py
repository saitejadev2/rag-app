from app.retrieval.retriever import Retriever
from app.generation.generator import Generator


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def query(self, question: str, k: int = 3):
        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            question,
            k=k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Build context for the LLM
        context_parts = []

        for document, metadata in zip(documents, metadatas):

            source = metadata.get("source", "Unknown")
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
            "sources": metadatas
        }