import os
import faiss
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from app.models.data_models import ChunkMetadata

class DenseRetriever:
    """FAISS In-Memory Dense Vector Retriever."""
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.metadata_store: List[ChunkMetadata] = []

    def build_index(self, chunks: List[ChunkMetadata]):
        self.metadata_store = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine similarity when normalized
        self.index.add(np.array(embeddings, dtype=np.float32))

    def load_index(self, index_path: str, metadata_list: List[Dict]):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.metadata_store = [ChunkMetadata(**m) for m in metadata_list]
        else:
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")

    def save_index(self, index_path: str):
        if self.index is not None:
            faiss.write_index(self.index, index_path)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[ChunkMetadata, float]]:
        if self.index is None or not self.metadata_store:
            return []
        
        query_vector = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vector, dtype=np.float32), top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.metadata_store):
                meta = self.metadata_store[idx]
                results.append((meta, float(score)))
                
        return results
