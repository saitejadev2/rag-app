from pathlib import Path

from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"


embedder = Embedder()

vector_store = VectorStore(
    persist_directory=str(CHROMA_DIR)
)

retriever = Retriever(
    vector_store=vector_store,
    embedder=embedder
)


question = input("Ask a question: ")

results = retriever.retrieve(
    question,
    k=5
)

print("\nRetrieved chunks:\n")

for i, (document, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ),
    start=1
):
    print(f"--- Result {i} ---")
    print(f"Source: {metadata.get('source')}")
    print(f"Page: {metadata.get('page', 'N/A')}")
    print(f"Chunk: {metadata.get('chunk_id')}")
    print(f"Distance: {distance:.4f}")
    print(f"Text: {document[:300]}")
    print()