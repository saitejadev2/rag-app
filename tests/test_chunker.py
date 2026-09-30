from pathlib import Path

from ingestion.loader import load_document
from ingestion.chunker import recursive_split


BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "documents" / "fastapi.txt"

text = load_document(str(file_path))

chunks = recursive_split(
    text,
    chunk_size=200,
    chunk_overlap=50
)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)
    print(f"Length: {len(chunk)}")