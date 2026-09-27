from pathlib import Path

from app.evaluation.retrieval_evaluator import RetrievalEvaluator


BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "data" / "chroma"

QUESTIONS_FILE = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "questions.json"
)


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
            f"Expected: "
            f"{result['expected_source']}"
        )

        print(
            f"Expected answer: "
            f"{result['expected_answer']}"
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

        print(
            f"Retrieved: "
            f"{result['retrieved_sources']}"
        )

        # -----------------------------------------------------
        # Citation information
        # -----------------------------------------------------

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

        if result["citation_precision"] is not None:

            print(
                f"Citation precision: "
                f"{result['citation_precision']:.2%}"
            )

        else:

            print(
                "Citation precision: N/A"
            )

        # -----------------------------------------------------
        # Retrieval metrics
        # -----------------------------------------------------

        if not result["is_unanswerable"]:

            print(
                f"Hit@1: "
                f"{result['hit@1']}"
            )

            print(
                f"Hit@3: "
                f"{result['hit@3']}"
            )

            print(
                f"Hit@5: "
                f"{result['hit@5']}"
            )

        else:

            print(
                "Retrieval metrics: "
                "N/A (unanswerable question)"
            )

        print()


    # =========================================================
    # Separate answerable / unanswerable questions
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

    total_answerable = len(
        answerable_results
    )

    total_unanswerable = len(
        unanswerable_results
    )


    # =========================================================
    # Retrieval metrics
    # =========================================================

    hit_at_1 = sum(
        result["hit@1"]
        for result in answerable_results
    )

    hit_at_3 = sum(
        result["hit@3"]
        for result in answerable_results
    )

    hit_at_5 = sum(
        result["hit@5"]
        for result in answerable_results
    )


    # =========================================================
    # Answer metrics
    # =========================================================

    answer_correct = sum(
        result["answer_correct"]
        for result in answerable_results
    )

    abstention_correct = sum(
        result["answer_correct"]
        for result in unanswerable_results
    )


    # =========================================================
    # Aggregate retrieval + answer metrics
    # =========================================================

    print(
        "\n================ AGGREGATE METRICS ================\n"
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
            f"Hit@1: "
            f"{hit_at_1}/{total_answerable} "
            f"({hit_at_1 / total_answerable:.2%})"
        )

        print(
            f"Hit@3: "
            f"{hit_at_3}/{total_answerable} "
            f"({hit_at_3 / total_answerable:.2%})"
        )

        print(
            f"Hit@5: "
            f"{hit_at_5}/{total_answerable} "
            f"({hit_at_5 / total_answerable:.2%})"
        )

        print(
            f"Answer accuracy: "
            f"{answer_correct}/{total_answerable} "
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

    else:

        print(
            "Citation precision: N/A"
        )

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
        f"{total_grounding_evaluated}/{len(results)}"
    )

    print(
        f"Grounded answers: "
        f"{total_grounded}/{total_grounding_evaluated}"
    )

    if total_grounding_evaluated > 0:

        grounding_accuracy = (
            total_grounded
            / total_grounding_evaluated
        )

        print(
            f"Grounding accuracy: "
            f"{grounding_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()