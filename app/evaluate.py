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

    print("\n================ RESULTS ================\n")

    for result in results:

        print(
            f"Question: {result['question']}"
        )

        print(
            f"Expected: {result['expected_source']}"
        )

        print(
            f"Retrieved: {result['retrieved_sources']}"
        )

        print(
            f"Hit@1: {result['hit@1']}"
        )

        print(
            f"Hit@3: {result['hit@3']}"
        )

        print(
            f"Hit@5: {result['hit@5']}"
        )

        print()


if __name__ == "__main__":
    main()