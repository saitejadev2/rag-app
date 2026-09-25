from pathlib import Path
from app.evaluation.retrieval_evaluator import RetrievalEvaluator


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
QUESTIONS_FILE = BASE_DIR / "data" / "evaluation" / "questions.json"


def main():
    evaluator = RetrievalEvaluator(chroma_dir=CHROMA_DIR)

    results = evaluator.evaluate(
        questions_file=QUESTIONS_FILE,
        conversation_id="evaluation"
    )

    print("\n================ RESULTS ================\n")

    for result in results:
        print(f"Question: {result['question']}")
        print(f"Expected: {result['expected_source']}")
        print(f"Expected answer: {result['expected_answer']}")
        print(f"Generated answer: {result['generated_answer']}")
        print(f"Answer correct: {result['answer_correct']}")
        print(f"Reason: {result['answer_reason']}")
        print(f"Retrieved: {result['retrieved_sources']}")

        if not result["is_unanswerable"]:
            print(f"Hit@1: {result['hit@1']}")
            print(f"Hit@3: {result['hit@3']}")
            print(f"Hit@5: {result['hit@5']}")
        else:
            print("Retrieval metrics: N/A (unanswerable question)")

        print()

    # Aggregate metrics
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

    total_answerable = len(answerable_results)
    total_unanswerable = len(unanswerable_results)

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

    answer_correct = sum(
        result["answer_correct"]
        for result in answerable_results
    )

    abstention_correct = sum(
        result["answer_correct"]
        for result in unanswerable_results
    )

    print("\n================ AGGREGATE METRICS ================\n")

    print(f"Answerable questions: {total_answerable}")
    print(f"Unanswerable questions: {total_unanswerable}")

    if total_answerable > 0:
        print(
            f"Hit@1: {hit_at_1}/{total_answerable} "
            f"({hit_at_1 / total_answerable:.2%})"
        )

        print(
            f"Hit@3: {hit_at_3}/{total_answerable} "
            f"({hit_at_3 / total_answerable:.2%})"
        )

        print(
            f"Hit@5: {hit_at_5}/{total_answerable} "
            f"({hit_at_5 / total_answerable:.2%})"
        )

        print(
            f"Answer accuracy: {answer_correct}/{total_answerable} "
            f"({answer_correct / total_answerable:.2%})"
        )

    if total_unanswerable > 0:
        print(
            f"Abstention accuracy: "
            f"{abstention_correct}/{total_unanswerable} "
            f"({abstention_correct / total_unanswerable:.2%})"
        )


if __name__ == "__main__":
    main()