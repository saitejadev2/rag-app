import json
from pathlib import Path

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker

class RetrievalEvaluator:

    def __init__(self, chroma_dir: str = "data/chroma"):
        self.embedder = Embedder()

        self.vector_store = VectorStore(
            persist_directory=str(chroma_dir)
        )

        self.reranker = Reranker()

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            reranker=self.reranker
        )

    def evaluate(
        self,
        questions_file: str,
        conversation_id: str,
        k_values: list[int] = [1, 3, 5]
    ):
        # Load evaluation questions
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)

        max_k = max(k_values)

        results = []

        for item in questions:

            question = item["question"]
            expected_source = item["expected_source"]

            retrieved = self.retriever.retrieve(
                query=question,
                k=max_k,
                conversation_id=conversation_id
            )

            metadatas = retrieved["metadatas"][0]
            distances = retrieved["distances"][0]
            reranker_scores = retrieved.get(
                "reranker_scores",
                [[]]
            )[0]

            sources = [
                metadata["source"]
                for metadata in metadatas
            ]

            # Print retrieval details
            print(f"\nQuestion: {question}")
            print(f"Expected source: {expected_source}")

            print("\nRetrieved chunks:")

            for rank, (source, distance, reranker_score) in enumerate(
                zip(sources, distances, reranker_scores),
                start=1
            ):
                print(
                    f"{rank}. {source} "
                    f"(chroma_distance={distance:.4f}, "
                    f"reranker_score={reranker_score:.4f})"
                )

            # Calculate Hit@K
            question_result = {
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": sources,
                "distances": distances,
                "reranker_scores": reranker_scores
            }

            for k in k_values:
                question_result[f"hit@{k}"] = (
                    expected_source in sources[:k]
                )

            results.append(question_result)

        return results

    def print_results(self, results):

        print("\n================ RESULTS ================")

        for result in results:

            print(f"\nQuestion: {result['question']}")

            print(
                f"Expected: {result['expected_source']}"
            )

            print(
                f"Retrieved: {result['retrieved_sources']}"
            )

            print(
                f"Distances: "
                f"{result['distances']}"
            )

            for key in ["hit@1", "hit@3", "hit@5"]:

                if key in result:
                    print(
                        f"{key.upper()}: "
                        f"{result[key]}"
                    )