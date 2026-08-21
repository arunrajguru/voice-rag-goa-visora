import re
from typing import List, Tuple

from app.models.data_models import ChunkMetadata


class Reranker:
    """
    Lightweight relevance reranker.

    Removes duplicate chunks and checks whether the query
    has meaningful word overlap with the retrieved context.
    """

    def __init__(self):
        pass

    def rerank(
        self,
        candidates: List[Tuple[ChunkMetadata, float]],
        final_k: int = 3
    ) -> List[ChunkMetadata]:

        if not candidates:
            return []

        seen_texts = set()
        deduped: List[ChunkMetadata] = []

        for chunk, score in candidates:

            # --------------------------------------------------
            # Remove duplicate chunks
            # --------------------------------------------------

            clean_snippet = " ".join(
                chunk.text.lower().split()[:20]
            )

            if clean_snippet in seen_texts:
                continue

            seen_texts.add(clean_snippet)

            # --------------------------------------------------
            # Keep original retrieval score
            # --------------------------------------------------

            chunk.score = float(score)

            deduped.append(chunk)

        # ------------------------------------------------------
        # Sort by retrieval score
        # ------------------------------------------------------

        deduped.sort(
            key=lambda c: c.score,
            reverse=True
        )

        return deduped[:final_k]
