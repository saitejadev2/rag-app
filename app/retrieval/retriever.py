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

    def retrieve(
        self,
        query: str,
        k: int = 3,
        max_distance: float | None = None,
        conversation_id: str | None = None
    ):
        # ------------------------------------------------
        # Step 1: Vector search
        # ------------------------------------------------

        query_embedding = self.embedder.embed_query(query)

        # Retrieve more candidates than we ultimately need
        candidate_k = max(k * 3, 10)

        results = self.vector_store.search(
            query_embedding,
            k=candidate_k,
            conversation_id=conversation_id
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # ------------------------------------------------
        # Step 2: Optional distance filtering
        # ------------------------------------------------

        if max_distance is not None:

            filtered = [
                (document, metadata, distance)
                for document, metadata, distance
                in zip(documents, metadatas, distances)
                if distance <= max_distance
            ]

            documents = [item[0] for item in filtered]
            metadatas = [item[1] for item in filtered]
            distances = [item[2] for item in filtered]

        if not documents:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        # ------------------------------------------------
        # Step 3: Reranking
        # ------------------------------------------------

        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=k
        )

        reranked_documents = []
        reranked_metadatas = []
        reranked_distances = []
        reranked_scores = []

        for document, reranker_score in reranked:

            # Find the original chunk position
            index = documents.index(document)

            reranked_documents.append(document)
            reranked_metadatas.append(metadatas[index])
            reranked_distances.append(distances[index])
            reranked_scores.append(float(reranker_score))

        return {
            "documents": [reranked_documents],
            "metadatas": [reranked_metadatas],
            "distances": [reranked_distances],
            "reranker_scores": [reranked_scores]
        }