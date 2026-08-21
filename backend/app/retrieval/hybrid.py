from typing import List, Tuple, Dict

from app.models.data_models import ChunkMetadata
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever


class HybridRetriever:
    """
    Hybrid Dense + BM25 Retriever.

    Uses:
    - Dense cosine similarity
    - BM25 relevance
    - Weighted fusion

    Important:
    Dense similarity is kept on its original 0-1 scale.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.6
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha

    @staticmethod
    def _normalize_bm25(
        scored_items: List[Tuple[ChunkMetadata, float]]
    ) -> Dict[str, float]:

        if not scored_items:
            return {}

        scores = [
            score
            for _, score in scored_items
        ]

        max_score = max(scores)

        if max_score <= 0:
            return {
                item.chunk_id: 0.0
                for item, _ in scored_items
            }

        # BM25 is converted to 0-1 using the maximum
        # BM25 score for this query.
        return {
            item.chunk_id: max(0.0, score / max_score)
            for item, score in scored_items
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        alpha_override: float = None
    ) -> List[Tuple[ChunkMetadata, float]]:

        alpha = (
            alpha_override
            if alpha_override is not None
            else self.alpha
        )

        # --------------------------------------------------
        # Dense retrieval
        # --------------------------------------------------

        dense_results = self.dense.retrieve(
            query,
            top_k=top_k
        )

        # --------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------

        bm25_results = self.bm25.retrieve(
            query,
            top_k=top_k
        )

        # --------------------------------------------------
        # Normalize BM25 only
        # --------------------------------------------------

        norm_bm25 = self._normalize_bm25(
            bm25_results
        )

        # --------------------------------------------------
        # Build chunk map
        # --------------------------------------------------

        chunk_map: Dict[
            str,
            ChunkMetadata
        ] = {}

        for chunk, dense_score in dense_results:

            # Keep REAL cosine similarity
            chunk.dense_score = float(
                dense_score
            )

            chunk_map[
                chunk.chunk_id
            ] = chunk

        for chunk, bm25_score in bm25_results:

            chunk.bm25_score = float(
                norm_bm25.get(
                    chunk.chunk_id,
                    0.0
                )
            )

            if chunk.chunk_id not in chunk_map:
                chunk_map[
                    chunk.chunk_id
                ] = chunk

        # --------------------------------------------------
        # Hybrid fusion
        # --------------------------------------------------

        hybrid_scores = []

        for chunk_id, chunk in chunk_map.items():

            dense_score = float(
                getattr(
                    chunk,
                    "dense_score",
                    0.0
                )
            )

            bm25_score = float(
                getattr(
                    chunk,
                    "bm25_score",
                    0.0
                )
            )

            # Weighted combination
            hybrid_score = (
                alpha * dense_score
                +
                (1.0 - alpha) * bm25_score
            )

            chunk.score = float(
                hybrid_score
            )

            hybrid_scores.append(
                (
                    chunk,
                    float(hybrid_score)
                )
            )

        # --------------------------------------------------
        # Sort
        # --------------------------------------------------

        hybrid_scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return hybrid_scores[:top_k]
