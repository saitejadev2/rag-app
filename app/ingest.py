from pathlib import Path

from app.ingestion.service import IngestionService


BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma"


def ingest_all_documents():

    service = IngestionService(
        documents_dir=DOCUMENTS_DIR,
        chroma_dir=CHROMA_DIR
    )

    files = list(DOCUMENTS_DIR.glob("*.txt"))
    files += list(DOCUMENTS_DIR.glob("*.pdf"))

    if not files:
        print("No documents found.")
        return

    total_chunks = 0

    for file_path in files:

        result = service.ingest_file(file_path)

        print(
            f"Indexed {result['filename']} "
            f"({result['chunks']} chunks)"
        )

        total_chunks += result["chunks"]

    print("\n----------------------------")
    print("Ingestion complete!")
    print(f"Total chunks: {total_chunks}")
    print("----------------------------")


if __name__ == "__main__":
    ingest_all_documents()