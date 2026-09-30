from pathlib import Path

from ingestion.loader import load_document


BASE_DIR = Path(__file__).resolve().parent.parent

file_path = (
    BASE_DIR
    / "data"
    / "documents"
    / "sample.pdf"
)


documents = load_document(
    str(file_path)
)

print(
    f"Loaded {len(documents)} page(s)"
)

for document in documents:

    print("\n--------------------")

    print(
        "Metadata:",
        document.metadata
    )

    print(
        "Text:",
        document.text[:300]
    )