import pickle
import os
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from app.models.data_models import ChunkMetadata
from app.utils.text_cleaner import tokenize

class BM25Retriever:
    """BM25 Lexical Sparse Retriever."""
    def __init__(self):
        self.bm25 = None
        self.metadata_store: List[ChunkMetadata] = []

    def build_index(self, chunks: List[ChunkMetadata]):
        self.metadata_store = chunks
        corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def save_index(self, path: str):
        if self.bm25 is not None:
            with open(path, "wb") as f:
                pickle.dump((self.bm25, self.metadata_store), f)

    def load_index(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.bm25, self.metadata_store = pickle.load(f)
        else:
            raise FileNotFoundError(f"BM25 index file not found at {path}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[ChunkMetadata, float]]:
        if self.bm25 is None or not self.metadata_store:
            return []
        
        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
            
        raw_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = float(raw_scores[idx])
            if score > 0:
                results.append((self.metadata_store[idx], score))
                
        return results
