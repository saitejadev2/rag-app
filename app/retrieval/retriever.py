from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve_candidates(
        self,
        query: str,
        candidate_k: int = 10,
        max_distance: float | None = None,
        conversation_id: str | None = None
    ):
        """
        Stage 1:
        Retrieve candidate documents using vector similarity.
        No reranking happens here.
        """

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding,
            k=candidate_k,
            conversation_id=conversation_id
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if max_distance is not None:

            filtered = [
                (document, metadata, distance)
                for document, metadata, distance
                in zip(
                    documents,
                    metadatas,
                    distances
                )
                if distance <= max_distance
            ]

            documents = [
                item[0]
                for item in filtered
            ]

            metadatas = [
                item[1]
                for item in filtered
            ]

            distances = [
                item[2]
                for item in filtered
            ]

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances
        }

    def retrieve(
        self,
        query: str,
        k: int = 3,
        max_distance: float | None = None,
        conversation_id: str | None = None
    ):
        """
        Full retrieval pipeline:

        Vector retrieval
              ↓
        Cross-encoder reranking
              ↓
        Top-k results
        """

        candidate_k = max(k * 3, 10)

        candidates = self.retrieve_candidates(
            query=query,
            candidate_k=candidate_k,
            max_distance=max_distance,
            conversation_id=conversation_id
        )

        documents = candidates["documents"]
        metadatas = candidates["metadatas"]
        distances = candidates["distances"]

        if not documents:

            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "reranker_scores": [[]]
            }

        pairs = [
            (query, document)
            for document in documents
        ]

        scores = self.reranker.model.predict(pairs)

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda index: scores[index],
            reverse=True
        )

        ranked_indices = ranked_indices[:k]

        reranked_documents = [
            documents[index]
            for index in ranked_indices
        ]

        reranked_metadatas = [
            metadatas[index]
            for index in ranked_indices
        ]

        reranked_distances = [
            distances[index]
            for index in ranked_indices
        ]

        reranked_scores = [
            float(scores[index])
            for index in ranked_indices
        ]

        return {
            "documents": [reranked_documents],
            "metadatas": [reranked_metadatas],
            "distances": [reranked_distances],
            "reranker_scores": [reranked_scores]
        }