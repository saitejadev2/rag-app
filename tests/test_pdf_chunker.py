from pathlib import Path

from ingestion.loader import load_document
from ingestion.chunker import chunk_documents


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

chunks = chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=50
)

print(
    f"Created {len(chunks)} chunks"
)

for chunk in chunks:

    print("\n--------------------")

    print(
        "Metadata:",
        chunk.metadata
    )

    print(
        "Text:",
        chunk.text[:200]
    )