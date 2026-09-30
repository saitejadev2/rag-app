from pathlib import Path

from ingestion.loader import load_document
from ingestion.chunker import recursive_split
from ingestion.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever

from generation.generator import Generator
from rag.pipeline import RAGPipeline


BASE_DIR = Path(__file__).resolve().parent.parent

file_path = (
    BASE_DIR
    / "data"
    / "documents"
    / "fastapi.txt"
)


# -------------------------
# INGESTION
# -------------------------

text = load_document(
    str(file_path)
)

chunks = recursive_split(
    text,
    chunk_size=200,
    chunk_overlap=50
)

embedder = Embedder()

embeddings = embedder.embed_documents(
    chunks
)


# -------------------------
# VECTOR STORE
# -------------------------

store = VectorStore(
    persist_directory=str(
        BASE_DIR / "data" / "chroma"
    )
)

metadatas = [
    {
        "source": file_path.name,
        "chunk_id": i
    }
    for i in range(len(chunks))
]


# -------------------------
# RETRIEVER
# -------------------------

retriever = Retriever(
    vector_store=store,
    embedder=embedder
)


# -------------------------
# GENERATOR
# -------------------------

generator = Generator()


# -------------------------
# RAG PIPELINE
# -------------------------

rag = RAGPipeline(
    retriever=retriever,
    generator=generator
)


# -------------------------
# QUERY
# -------------------------

question = "How does FastAPI handle authentication?"

answer = rag.query(
    question,
    k=3
)

print("\nANSWER")
print("=" * 50)
print(answer)