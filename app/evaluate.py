
from pathlib import Path

from app.evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)


BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = (
    BASE_DIR
    / "data"
    / "chroma"
)

QUESTIONS_FILE = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "questions.json"
)


def average_metric(
    results,
    metric_group,
    metric_name,
    k
):
    values = []

    key = f"{metric_name}@{k}"

    for result in results:

        if result["is_unanswerable"]:
            continue

        value = result[
            metric_group
        ].get(key)

        if value is not None:
            values.append(value)

    if not values:
        return None

    return sum(values) / len(values)


def reranking_comparison(results, k):
    """
    Compare vector retrieval and reranked retrieval
    on a per-question basis.

    A question is classified as:

    - improved: reranking increased chunk recall
    - same: reranking produced the same recall
    - worse: reranking decreased chunk recall
    """

    improved = []
    same = []
    worse = []

    vector_key = f"vector_chunk_recall@{k}"
    reranked_key = f"reranked_chunk_recall@{k}"

    for result in results:

        if result["is_unanswerable"]:
            continue

        vector_recall = result[
            "vector_chunk_recalls"
        ].get(vector_key)

        reranked_recall = result[
            "reranked_chunk_recalls"
        ].get(reranked_key)

        if vector_recall is None or reranked_recall is None:
            continue

        question = result["question"]

        comparison = {
            "question": question,
            "vector": vector_recall,
            "reranked": reranked_recall,
        }

        if reranked_recall > vector_recall:
            improved.append(comparison)

        elif reranked_recall < vector_recall:
            worse.append(comparison)

        else:
            same.append(comparison)

    total = (
        len(improved)
        + len(same)
        + len(worse)
    )

    return {
        "improved": improved,
        "same": same,
        "worse": worse,
        "total": total,
    }


def main():

    evaluator = RetrievalEvaluator(
        chroma_dir=CHROMA_DIR
    )

    results = evaluator.evaluate(
        questions_file=QUESTIONS_FILE,
        conversation_id="evaluation"
    )

    # =========================================================
    # Detailed results
    # =========================================================

    print(
        "\n================ RESULTS ================\n"
    )

    for result in results:

        print(
            f"Question: "
            f"{result['question']}"
        )

        print(
            f"Expected chunks: "
            f"{result['expected_chunks']}"
        )

        print(
            f"Generated answer: "
            f"{result['generated_answer']}"
        )

        print(
            f"Answer correct: "
            f"{result['answer_correct']}"
        )

        print(
            f"Reason: "
            f"{result['answer_reason']}"
        )

        if not result["is_unanswerable"]:

            print(
                "Vector retrieval:"
            )

            for k in [1, 3, 5]:

                value = result[
                    "vector_chunk_recalls"
                ][
                    f"vector_chunk_recall@{k}"
                ]

                print(
                    f"  Chunk Recall@{k}: "
                    f"{value:.2%}"
                )

            print(
                "After reranking:"
            )

            for k in [1, 3, 5]:

                value = result[
                    "reranked_chunk_recalls"
                ][
                    f"reranked_chunk_recall@{k}"
                ]

                print(
                    f"  Chunk Recall@{k}: "
                    f"{value:.2%}"
                )

        else:

            print(
                "Retrieval metrics: "
                "N/A (unanswerable question)"
            )

        print(
            f"Citations: "
            f"{result['citation_count']}"
        )

        print(
            f"Valid citations: "
            f"{result['valid_citations']}"
        )

        print(
            f"Invalid citations: "
            f"{result['invalid_citations']}"
        )

        if result["grounded"] is not None:

            print(
                f"Grounded: "
                f"{result['grounded']}"
            )

            print(
                f"Grounding reason: "
                f"{result['grounding_reason']}"
            )

        else:

            print(
                "Grounded: N/A"
            )

        print()


    # =========================================================
    # Split answerable / unanswerable
    # =========================================================

    answerable_results = [
        result
        for result in results
        if not result["is_unanswerable"]
    ]

    unanswerable_results = [
        result
        for result in results
        if result["is_unanswerable"]
    ]


    # =========================================================
    # Retrieval comparison
    # =========================================================

    print(
        "\n================ RETRIEVAL COMPARISON ================\n"
    )

    for k in [1, 3, 5]:

        vector_value = average_metric(
            results,
            "vector_chunk_recalls",
            "vector_chunk_recall",
            k
        )

        reranked_value = average_metric(
            results,
            "reranked_chunk_recalls",
            "reranked_chunk_recall",
            k
        )

        print(
            f"Chunk Recall@{k}:"
        )

        print(
            f"  Vector retrieval: "
            f"{vector_value:.2%}"
        )

        print(
            f"  After reranking: "
            f"{reranked_value:.2%}"
        )

        improvement = (
            reranked_value
            - vector_value
        )

        print(
            f"  Improvement: "
            f"{improvement:+.2%}"
        )

        print()


    # =========================================================
    # Per-question reranking analysis
    # =========================================================

    print(
        "\n================ RERANKING EFFECT BY QUESTION ================\n"
    )

    for k in [1, 3, 5]:

        comparison = reranking_comparison(
            results,
            k
        )

        total = comparison["total"]

        print(
            f"Recall@{k}"
        )

        print(
            "-" * 30
        )

        print(
            f"Improved: "
            f"{len(comparison['improved'])}/{total}"
        )

        print(
            f"Same:     "
            f"{len(comparison['same'])}/{total}"
        )

        print(
            f"Worse:    "
            f"{len(comparison['worse'])}/{total}"
        )

        if total > 0:

            print(
                f"Improvement rate: "
                f"{len(comparison['improved']) / total:.2%}"
            )

            print(
                f"Same rate:        "
                f"{len(comparison['same']) / total:.2%}"
            )

            print(
                f"Worse rate:       "
                f"{len(comparison['worse']) / total:.2%}"
            )

        # -----------------------------------------------------
        # Questions where reranking improved retrieval
        # -----------------------------------------------------

        if comparison["improved"]:

            print()
            print(
                "Improved questions:"
            )

            for item in comparison["improved"]:

                print(
                    f"  - {item['question']}"
                )

                print(
                    f"    Vector: "
                    f"{item['vector']:.2%}"
                )

                print(
                    f"    Reranked: "
                    f"{item['reranked']:.2%}"
                )

        # -----------------------------------------------------
        # Questions where reranking made retrieval worse
        # -----------------------------------------------------

        if comparison["worse"]:

            print()
            print(
                "Worse questions:"
            )

            for item in comparison["worse"]:

                print(
                    f"  - {item['question']}"
                )

                print(
                    f"    Vector: "
                    f"{item['vector']:.2%}"
                )

                print(
                    f"    Reranked: "
                    f"{item['reranked']:.2%}"
                )

        print()


    # =========================================================
    # Answer metrics
    # =========================================================

    total_answerable = len(
        answerable_results
    )

    total_unanswerable = len(
        unanswerable_results
    )

    answer_correct = sum(
        result["answer_correct"]
        for result in answerable_results
    )

    abstention_correct = sum(
        result["answer_correct"]
        for result in unanswerable_results
    )

    print(
        "\n================ ANSWER METRICS ================\n"
    )

    print(
        f"Answerable questions: "
        f"{total_answerable}"
    )

    print(
        f"Unanswerable questions: "
        f"{total_unanswerable}"
    )

    if total_answerable > 0:

        print(
            f"Answer accuracy: "
            f"{answer_correct}/"
            f"{total_answerable} "
            f"({answer_correct / total_answerable:.2%})"
        )

    if total_unanswerable > 0:

        print(
            f"Abstention accuracy: "
            f"{abstention_correct}/"
            f"{total_unanswerable} "
            f"({abstention_correct / total_unanswerable:.2%})"
        )


    # =========================================================
    # Citation metrics
    # =========================================================

    total_citations = sum(
        result["citation_count"]
        for result in results
    )

    total_valid_citations = sum(
        result["valid_citations"]
        for result in results
    )

    total_invalid_citations = sum(
        result["invalid_citations"]
        for result in results
    )

    questions_with_citations = sum(
        result["has_citation"]
        for result in results
    )

    print(
        "\n================ CITATION METRICS ================\n"
    )

    print(
        f"Questions with citations: "
        f"{questions_with_citations}/{len(results)}"
    )

    print(
        f"Total citations: "
        f"{total_citations}"
    )

    print(
        f"Valid citations: "
        f"{total_valid_citations}"
    )

    print(
        f"Invalid citations: "
        f"{total_invalid_citations}"
    )

    if total_citations > 0:

        citation_precision = (
            total_valid_citations
            / total_citations
        )

        print(
            f"Citation precision: "
            f"{citation_precision:.2%}"
        )


    # =========================================================
    # Grounding metrics
    # =========================================================

    grounded_results = [
        result
        for result in results
        if result["grounded"] is not None
    ]

    total_grounded = sum(
        result["grounded"]
        for result in grounded_results
    )

    total_grounding_evaluated = len(
        grounded_results
    )

    print(
        "\n================ GROUNDING METRICS ================\n"
    )

    print(
        f"Grounding evaluated: "
        f"{total_grounding_evaluated}/"
        f"{len(results)}"
    )

    if total_grounding_evaluated > 0:

        print(
            f"Grounded answers: "
            f"{total_grounded}/"
            f"{total_grounding_evaluated}"
        )

        print(
            f"Grounding accuracy: "
            f"{total_grounded / total_grounding_evaluated:.2%}"
        )


if __name__ == "__main__":
    main()
