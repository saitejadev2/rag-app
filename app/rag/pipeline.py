from retrieval.retriever import Retriever
from generation.generator import Generator


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
        k: int = 3
    ) -> str:

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            question,
            k=k
        )

        # Extract retrieved text
        documents = results["documents"][0]

        # Combine chunks into context
        context = "\n\n".join(documents)

        # Generate answer
        answer = self.generator.generate(
            query=question,
            context=context
        )

        return answer