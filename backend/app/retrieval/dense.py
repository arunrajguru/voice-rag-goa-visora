"""
Lightweight dense-style retriever for low-memory deployments.

This version intentionally does NOT use:
- torch
- sentence-transformers
- faiss

It provides the same DenseRetriever interface expected by
HybridRetriever, but uses lightweight token-frequency cosine similarity.
"""

import math
import os
import pickle
import re
from collections import Counter
from typing import List, Tuple, Dict

from app.models.data_models import ChunkMetadata


class DenseRetriever:
    """
    Lightweight similarity retriever.

    This keeps the same public interface as the original FAISS +
    SentenceTransformer retriever so HybridRetriever does not need
    to be changed.
    """

    def __init__(self, embedding_model_name: str = ""):
        # Kept only for compatibility with the existing configuration.
        self.embedding_model_name = embedding_model_name

        self.index = None
        self.metadata_store: List[ChunkMetadata] = []
        self.document_vectors: List[Dict[str, float]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Convert text into lightweight normalized tokens."""
        if not text:
            return []

        text = text.lower()
        return re.findall(r"\b[a-zA-Z0-9]+\b", text)

    @classmethod
    def _vectorize(cls, text: str) -> Dict[str, float]:
        """
        Create a normalized term-frequency vector.

        This is intentionally simple and memory efficient.
        """
        tokens = cls._tokenize(text)

        if not tokens:
            return {}

        counts = Counter(tokens)
        total = float(len(tokens))

        return {
            token: count / total
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine_similarity(
        vector_a: Dict[str, float],
        vector_b: Dict[str, float],
    ) -> float:
        """Calculate cosine similarity between sparse dictionaries."""

        if not vector_a or not vector_b:
            return 0.0

        # Iterate through the smaller dictionary.
        if len(vector_a) > len(vector_b):
            vector_a, vector_b = vector_b, vector_a

        dot_product = sum(
            value * vector_b.get(token, 0.0)
            for token, value in vector_a.items()
        )

        norm_a = math.sqrt(
            sum(value * value for value in vector_a.values())
        )

        norm_b = math.sqrt(
            sum(value * value for value in vector_b.values())
        )

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def build_index(self, chunks: List[ChunkMetadata]):
        """
        Build the lightweight in-memory similarity index.
        """

        self.metadata_store = list(chunks)

        self.document_vectors = [
            self._vectorize(chunk.text)
            for chunk in self.metadata_store
        ]

        # Keep a non-None marker for compatibility with existing code.
        self.index = True

    def load_index(
        self,
        index_path: str,
        metadata_list: List[Dict],
    ):
        """
        Compatibility method.

        The old implementation loaded a FAISS index from disk.
        This lightweight version only needs the metadata and rebuilds
        the small similarity index from the text.
        """

        if not metadata_list:
            raise ValueError("No metadata available to build retrieval index.")

        self.metadata_store = [
            ChunkMetadata(**metadata)
            for metadata in metadata_list
        ]

        self.document_vectors = [
            self._vectorize(chunk.text)
            for chunk in self.metadata_store
        ]

        self.index = True

    def save_index(self, index_path: str):
        """
        Save the lightweight index.

        This is optional and mainly exists for compatibility with
        existing indexing scripts.
        """

        if not self.metadata_store:
            return

        directory = os.path.dirname(index_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        data = {
            "metadata": [
                chunk.model_dump()
                if hasattr(chunk, "model_dump")
                else chunk.dict()
                for chunk in self.metadata_store
            ],
            "vectors": self.document_vectors,
        }

        with open(index_path, "wb") as file:
            pickle.dump(data, file)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Retrieve the most similar chunks.
        """

        if not self.metadata_store:
            return []

        query_vector = self._vectorize(query)

        if not query_vector:
            return []

        scored_results = []

        for chunk, document_vector in zip(
            self.metadata_store,
            self.document_vectors,
        ):
            score = self._cosine_similarity(
                query_vector,
                document_vector,
            )

            if score > 0:
                scored_results.append((chunk, float(score)))

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_results[:top_k]
