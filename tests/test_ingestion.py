from pathlib import Path

from ingestion.loader import load_document
from ingestion.chunker import recursive_split
from ingestion.embedder import Embedder


BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "documents" / "fastapi.txt"


# 1. Load
text = load_document(str(file_path))

# 2. Chunk
chunks = recursive_split(
    text,
    chunk_size=200,
    chunk_overlap=50
)

print(f"Number of chunks: {len(chunks)}")


# 3. Embed
embedder = Embedder()

embeddings = embedder.embed_documents(chunks)

print(f"Embedding shape: {embeddings.shape}")