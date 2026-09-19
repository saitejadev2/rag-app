import chromadb


class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma"):
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
        ids = []

        for metadata in metadatas:
            source = metadata["source"]
            page = metadata.get("page", "na")
            chunk_id = metadata["chunk_id"]

            chunk_id_string = (
                f"{source}-page-{page}-chunk-{chunk_id}"
            )

            ids.append(chunk_id_string)

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def delete_by_source(self, source: str):
        self.collection.delete(
            where={
                "source": source
            }
        )

    def search(self, query_embedding, k: int = 3):
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        return results

    def size(self):
        return self.collection.count()

    def get_sources(self):
        results = self.collection.get(
            include=["metadatas"]
        )

        sources = set()

        for metadata in results["metadatas"]:
            if metadata and "source" in metadata:
                sources.add(metadata["source"])

        return sorted(sources)