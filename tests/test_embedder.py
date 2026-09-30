from ingestion.embedder import Embedder


embedder = Embedder()

texts = [
    "FastAPI is a Python web framework.",
    "PostgreSQL is a relational database.",
    "FastAPI supports API authentication."
]

embeddings = embedder.embed_documents(texts)

print("Shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])