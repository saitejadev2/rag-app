import json

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.rag.pipeline import RAGPipeline

from app.evaluation.answer_evaluator import AnswerEvaluator
from app.evaluation.citation_evaluator import CitationEvaluator
from app.evaluation.grounding_evaluator import GroundingEvaluator


class RetrievalEvaluator:

    def __init__(
        self,
        chroma_dir,
        k_values=None
    ):
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

        self.citation_evaluator = (
            CitationEvaluator()
        )

        self.grounding_evaluator = (
            GroundingEvaluator()
        )

        self.k_values = (
            k_values
            if k_values is not None
            else [1, 3, 5]
        )

    # =========================================================
    # Chunk recall
    # =========================================================

    def chunk_recall(
        self,
        expected_chunks,
        retrieved_metadatas,
        k
    ):
        if not expected_chunks:
            return None

        retrieved_chunks = [
            metadata.get("chunk_id")
            for metadata in retrieved_metadatas[:k]
        ]

        expected_set = set(expected_chunks)
        retrieved_set = set(retrieved_chunks)

        relevant_retrieved = (
            expected_set & retrieved_set
        )

        return (
            len(relevant_retrieved)
            / len(expected_set)
        )

    # =========================================================
    # Evaluate
    # =========================================================

    def evaluate(
        self,
        questions_file,
        conversation_id=None
    ):

        with open(
            questions_file,
            "r",
            encoding="utf-8"
        ) as file:
            questions = json.load(file)

        max_k = max(self.k_values)

        results = []

        for index, item in enumerate(
            questions,
            start=1
        ):

            question = item["question"]

            expected_answer = item[
                "expected_answer"
            ]

            expected_chunks = item.get(
                "expected_chunks",
                []
            )

            is_unanswerable = item[
                "is_unanswerable"
            ]

            print(
                f"\n{'=' * 70}"
            )

            print(
                f"Question {index}/{len(questions)}"
            )

            print(
                f"{question}"
            )

            # =================================================
            # Stage 1: Vector retrieval
            # =================================================

            candidates = (
                self.retriever.retrieve_candidates(
                    query=question,
                    candidate_k=max(
                        max_k * 3,
                        10
                    ),
                    conversation_id=conversation_id
                )
            )

            candidate_documents = candidates[
                "documents"
            ]

            candidate_metadatas = candidates[
                "metadatas"
            ]

            candidate_distances = candidates[
                "distances"
            ]

            # =================================================
            # Vector retrieval metrics
            # =================================================

            vector_recalls = {}

            for k in self.k_values:

                vector_recalls[
                    f"vector_chunk_recall@{k}"
                ] = self.chunk_recall(
                    expected_chunks,
                    candidate_metadatas,
                    k
                )

            # =================================================
            # Stage 2: Cross-encoder reranking
            # =================================================

            if candidate_documents:

                pairs = [
                    (
                        question,
                        document
                    )
                    for document in candidate_documents
                ]

                scores = (
                    self.reranker.model.predict(
                        pairs
                    )
                )

                ranked_indices = sorted(
                    range(
                        len(candidate_documents)
                    ),
                    key=lambda i: scores[i],
                    reverse=True
                )

                reranked_documents = [
                    candidate_documents[i]
                    for i in ranked_indices[:max_k]
                ]

                reranked_metadatas = [
                    candidate_metadatas[i]
                    for i in ranked_indices[:max_k]
                ]

                reranked_distances = [
                    candidate_distances[i]
                    for i in ranked_indices[:max_k]
                ]

                reranked_scores = [
                    float(scores[i])
                    for i in ranked_indices[:max_k]
                ]

            else:

                reranked_documents = []
                reranked_metadatas = []
                reranked_distances = []
                reranked_scores = []

            # =================================================
            # Reranked metrics
            # =================================================

            reranked_recalls = {}

            for k in self.k_values:

                reranked_recalls[
                    f"reranked_chunk_recall@{k}"
                ] = self.chunk_recall(
                    expected_chunks,
                    reranked_metadatas,
                    k
                )

            # =================================================
            # Build retrieval result for RAG pipeline
            # =================================================

            retrieved = {
                "documents": [
                    reranked_documents
                ],
                "metadatas": [
                    reranked_metadatas
                ],
                "distances": [
                    reranked_distances
                ],
                "reranker_scores": [
                    reranked_scores
                ]
            }

            # =================================================
            # Generate answer
            # =================================================

            rag_result = self.rag_pipeline.query(
                question=question,
                k=max_k,
                conversation_id=conversation_id,
                retrieved=retrieved
            )

            generated_answer = rag_result[
                "answer"
            ]

            # =================================================
            # Answer evaluation
            # =================================================

            answer_result = (
                self.answer_evaluator.evaluate(
                    question=question,
                    expected_answer=expected_answer,
                    generated_answer=generated_answer
                )
            )

            # =================================================
            # Citation evaluation
            # =================================================

            citation_result = (
                self.citation_evaluator.evaluate(
                    answer=generated_answer,
                    retrieved_sources=rag_result[
                        "sources"
                    ]
                )
            )

            # =================================================
            # Reconstruct cited chunks
            # =================================================

            cited_chunks = []

            citations = citation_result[
                "citations"
            ]

            for citation in citations:

                for document, metadata in zip(
                    reranked_documents,
                    reranked_metadatas
                ):

                    if (
                        citation["source"]
                        != metadata.get("source")
                    ):
                        continue

                    if (
                        citation["chunk_id"]
                        != metadata.get("chunk_id")
                    ):
                        continue

                    if "page" in citation:

                        if (
                            citation["page"]
                            != metadata.get("page")
                        ):
                            continue

                    cited_chunks.append(
                        {
                            **metadata,
                            "text": document
                        }
                    )

                    break

            # =================================================
            # Grounding evaluation
            # =================================================

            grounding_result = (
                self.grounding_evaluator.evaluate(
                    question=question,
                    answer=generated_answer,
                    cited_chunks=cited_chunks
                )
            )

            # =================================================
            # Print retrieval information
            # =================================================

            print(
                "\nVector retrieval:"
            )

            for rank, metadata in enumerate(
                candidate_metadatas[:max_k],
                start=1
            ):

                print(
                    f"  {rank}. "
                    f"chunk={metadata.get('chunk_id')} "
                    f"distance="
                    f"{candidate_distances[rank - 1]:.4f}"
                )

            print(
                "\nAfter reranking:"
            )

            for rank, metadata in enumerate(
                reranked_metadatas,
                start=1
            ):

                print(
                    f"  {rank}. "
                    f"chunk={metadata.get('chunk_id')} "
                    f"score="
                    f"{reranked_scores[rank - 1]:.4f}"
                )

            # =================================================
            # Store result
            # =================================================

            result = {
                "question": question,
                "expected_answer": expected_answer,
                "expected_source": item.get(
                    "expected_source"
                ),
                "expected_chunks": expected_chunks,
                "generated_answer": generated_answer,

                "is_unanswerable": (
                    is_unanswerable
                ),

                "vector_chunk_recalls":
                    vector_recalls,

                "reranked_chunk_recalls":
                    reranked_recalls,

                "answer_correct":
                    answer_result["correct"],

                "answer_reason":
                    answer_result["reason"],

                "retrieved_sources": [
                    metadata.get("source")
                    for metadata in reranked_metadatas
                ],

                "citation_count":
                    citation_result[
                        "citation_count"
                    ],

                "valid_citations":
                    citation_result[
                        "valid_citations"
                    ],

                "invalid_citations":
                    citation_result[
                        "invalid_citations"
                    ],

                "citation_precision":
                    citation_result[
                        "citation_precision"
                    ],

                "has_citation":
                    citation_result[
                        "has_citation"
                    ],

                "grounded":
                    grounding_result.get(
                        "grounded"
                    ),

                "grounding_reason":
                    grounding_result.get(
                        "reason"
                    )
            }

            results.append(result)

        return results