import chromadb


class VectorStore:

    def __init__(
        self,
        persist_directory: str = "data/chroma"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

    def add(
        self,
        chunks: list[str],
        embeddings,
        metadatas: list[dict]
    ):
        ids = [
            f"{metadata['source']}-{metadata['chunk_id']}"
            for metadata in metadatas
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding,
        k: int = 3
    ):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        return results

    def size(self):
        return self.collection.count()