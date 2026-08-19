from typing import List, Tuple, Dict
from app.models.data_models import ChunkMetadata
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever

class HybridRetriever:
    """Hybrid Dense + BM25 Retriever with score normalization and weighted fusion."""
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, alpha: float = 0.6):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha

    def _normalize_scores(self, scored_items: List[Tuple[ChunkMetadata, float]]) -> Dict[str, float]:
        if not scored_items:
            return {}
        scores = [score for _, score in scored_items]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return {item.chunk_id: 1.0 for item, _ in scored_items}
        return {item.chunk_id: (score - min_s) / (max_s - min_s) for item, score in scored_items}

    def retrieve(self, query: str, top_k: int = 15, alpha_override: float = None) -> List[Tuple[ChunkMetadata, float]]:
        alpha = alpha_override if alpha_override is not None else self.alpha
        
        dense_results = self.dense.retrieve(query, top_k=top_k)
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        
        norm_dense = self._normalize_scores(dense_results)
        norm_bm25 = self._normalize_scores(bm25_results)
        
        chunk_map: Dict[str, ChunkMetadata] = {}
        for chunk, score in dense_results:
            chunk.dense_score = norm_dense.get(chunk.chunk_id, 0.0)
            chunk_map[chunk.chunk_id] = chunk
            
        for chunk, score in bm25_results:
            chunk.bm25_score = norm_bm25.get(chunk.chunk_id, 0.0)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        hybrid_scores: List[Tuple[ChunkMetadata, float]] = []
        for chunk_id, chunk in chunk_map.items():
            d_s = norm_dense.get(chunk_id, 0.0)
            b_s = norm_bm25.get(chunk_id, 0.0)
            h_score = alpha * d_s + (1.0 - alpha) * b_s
            chunk.score = float(h_score)
            hybrid_scores.append((chunk, float(h_score)))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_k]
