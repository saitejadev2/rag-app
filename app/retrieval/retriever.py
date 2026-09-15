from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 3,
        max_distance: float | None = None
    ):
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding,
            k=k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # If no threshold is provided,
        # return all top-k results.
        if max_distance is None:
            return results

        # Keep only sufficiently similar chunks
        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):
            if distance <= max_distance:
                filtered_documents.append(document)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)

        return {
            "documents": [filtered_documents],
            "metadatas": [filtered_metadatas],
            "distances": [filtered_distances]
        }