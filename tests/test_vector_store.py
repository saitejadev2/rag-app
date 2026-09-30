from pathlib import Path

from ingestion.loader import load_document
from ingestion.chunker import recursive_split
from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "documents" / "fastapi.txt"


# 1. Load document
text = load_document(str(file_path))


# 2. Chunk document
chunks = recursive_split(
    text,
    chunk_size=200,
    chunk_overlap=50
)

print("Number of chunks:", len(chunks))


# 3. Generate embeddings
embedder = Embedder()

embeddings = embedder.embed_documents(chunks)

print("Embedding shape:", embeddings.shape)


# 4. Create vector store
store = VectorStore(
    persist_directory=str(BASE_DIR / "data" / "chroma")
)


# 5. Create metadata
metadatas = [
    {
        "source": file_path.name,
        "chunk_id": i
    }
    for i in range(len(chunks))
]


# 6. Store everything
store.add(
    chunks=chunks,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Vectors stored:", store.size())

query = "How does FastAPI handle authentication?"

query_embedding = embedder.embed_query(query)

results = store.search(
    query_embedding,
    k=3
)

print("\nRETRIEVED RESULTS")
print("=" * 50)

for document, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):
    print("\nDocument:")
    print(document)

    print("Metadata:")
    print(metadata)