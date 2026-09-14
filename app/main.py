from pathlib import Path

from ingestion.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever

from generation.generator import Generator

from rag.pipeline import RAGPipeline


BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "data" / "chroma"


# -------------------------
# Load existing components
# -------------------------

embedder = Embedder()

vector_store = VectorStore(
    persist_directory=str(CHROMA_DIR)
)

retriever = Retriever(
    vector_store=vector_store,
    embedder=embedder
)

generator = Generator()

rag = RAGPipeline(
    retriever=retriever,
    generator=generator
)


# -------------------------
# Ask question
# -------------------------

question = input("\nAsk a question: ")

answer = rag.query(
    question,
    k=3
)

print("\nAnswer:")
print(answer)