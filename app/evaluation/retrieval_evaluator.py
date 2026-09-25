import json

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker

from app.generation.generator import Generator
from app.rag.pipeline import RAGPipeline

from app.evaluation.answer_evaluator import AnswerEvaluator


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

        self.generator = Generator()

        self.rag_pipeline = RAGPipeline(
            retriever=self.retriever,
            generator=self.generator
        )

        self.answer_evaluator = AnswerEvaluator()

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
            expected_answer = item["expected_answer"]
            expected_source = item["expected_source"]
            is_unanswerable = item.get(
                "is_unanswerable",
                False
            )

            # -------------------------------------------------
            # 1. Retrieve and rerank ONCE
            # -------------------------------------------------

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

            # -------------------------------------------------
            # 2. Print retrieval details
            # -------------------------------------------------

            print(f"\nQuestion: {question}")
            print(f"Expected source: {expected_source}")

            print("\nRetrieved chunks:")

            for rank, (
                source,
                distance,
                reranker_score
            ) in enumerate(
                zip(
                    sources,
                    distances,
                    reranker_scores
                ),
                start=1
            ):
                print(
                    f"{rank}. {source} "
                    f"(chroma_distance={distance:.4f}, "
                    f"reranker_score={reranker_score:.4f})"
                )

            # -------------------------------------------------
            # 3. Generate answer using the SAME retrieval
            # -------------------------------------------------

            rag_result = self.rag_pipeline.query(
                question=question,
                k=3,
                conversation_id=conversation_id,
                retrieved=retrieved
            )

            generated_answer = rag_result["answer"]

            # -------------------------------------------------
            # 4. Evaluate generated answer
            # -------------------------------------------------

            answer_evaluation = self.answer_evaluator.evaluate(
                question=question,
                expected_answer=expected_answer,
                generated_answer=generated_answer
            )

            # -------------------------------------------------
            # 5. Store evaluation result
            # -------------------------------------------------

            question_result = {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "expected_source": expected_source,
                "retrieved_sources": sources,
                "distances": distances,
                "reranker_scores": reranker_scores,
                "answer_correct": answer_evaluation["correct"],
                "answer_reason": answer_evaluation["reason"],
                "is_unanswerable": is_unanswerable
            }

            # -------------------------------------------------
            # 6. Calculate Hit@K
            # -------------------------------------------------

            if not is_unanswerable:

                for k in k_values:
                    question_result[f"hit@{k}"] = (
                        expected_source in sources[:k]
                    )

            results.append(question_result)

        return results

    def print_results(self, results):

        print(
            "\n================ RESULTS ================\n"
        )

        for result in results:

            print(
                f"Question: {result['question']}"
            )

            print(
                f"Expected: "
                f"{result['expected_source']}"
            )

            print(
                f"Retrieved: "
                f"{result['retrieved_sources']}"
            )

            print(
                f"Distances: "
                f"{result['distances']}"
            )

            for key in [
                "hit@1",
                "hit@3",
                "hit@5"
            ]:

                if key in result:
                    print(
                        f"{key.upper()}: "
                        f"{result[key]}"
                    )

            print(
                f"Answer correct: "
                f"{result['answer_correct']}"
            )

            print(
                f"Generated answer: "
                f"{result['generated_answer']}"
            )

            if result["is_unanswerable"]:
                print(
                    "Retrieval metrics: "
                    "N/A (unanswerable question)"
                )

            print()