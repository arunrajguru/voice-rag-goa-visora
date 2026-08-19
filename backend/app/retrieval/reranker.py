from typing import List, Tuple
from app.models.data_models import ChunkMetadata

class Reranker:
    """Lightweight Cosine Reranker & Duplicate Removal.
    
    Why lightweight reranking:
    Heavy cross-encoder rerankers add 150-300 ms of latency per request, violating our <200 ms target.
    This lightweight reranker performs text overlap deduplication, length-normalized score adjustment,
    and returns top-K unique context passages.
    """
    def __init__(self):
        pass

    def rerank(self, candidates: List[Tuple[ChunkMetadata, float]], final_k: int = 3) -> List[ChunkMetadata]:
        if not candidates:
            return []

        seen_texts = set()
        deduped: List[ChunkMetadata] = []

        for chunk, score in candidates:
            # Clean preview to check for duplicate/near-duplicate chunk text
            clean_snippet = " ".join(chunk.text.lower().split()[:20])
            if clean_snippet in seen_texts:
                continue
            seen_texts.add(clean_snippet)
            
            # Subtle length adjustment boost for chunks that contain full context (>40 words)
            word_count = len(chunk.text.split())
            length_factor = min(1.1, 0.9 + (word_count / 200.0))
            chunk.score = float(score * length_factor)
            deduped.append(chunk)

        # Re-sort by adjusted final score
        deduped.sort(key=lambda c: c.score, reverse=True)
        return deduped[:final_k]
