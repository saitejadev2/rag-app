from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents


file_path = "data/documents/evaluation/evaluation.txt"

documents = load_document(file_path)

chunks = chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=50
)

print(f"\nTotal chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    preview = chunk.text[:250].replace("\n", " ")

    print(f"CHUNK {i}")
    print("-" * 50)
    print(preview)
    print()