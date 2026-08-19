import re
import numpy as np
from typing import List, Optional
from app.models.data_models import ChunkMetadata
from app.utils.text_cleaner import clean_text

class SemanticChunker:
    """Semantic boundary chunking using sentence embedding similarities.
    
    Why this exists:
    Groups sentences into topic-coherent passages by measuring cosine distance between consecutive sentences.
    Creates natural, contextually full chunks ideal for complex reasoning and semantic queries.
    """
    def __init__(self, similarity_threshold: float = 0.5, embedding_model = None):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, document_id: str, text: str, source: str = "MSMARCO-XI") -> List[ChunkMetadata]:
        cleaned = clean_text(text)
        if not cleaned:
            return []
        
        sentences = self._split_sentences(cleaned)
        if len(sentences) <= 1:
            return [ChunkMetadata(
                document_id=document_id,
                chunk_id=f"{document_id}_sem_0",
                strategy="semantic",
                position=0,
                text=cleaned,
                source=source
            )]

        # If embedding model is provided, compute adjacent sentence similarities
        if self.embedding_model is not None:
            embeddings = self.embedding_model.encode(sentences, normalize_embeddings=True)
            # Dot product of normalized vectors = cosine similarity
            similarities = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(sentences) - 1)]
        else:
            # Lexical Jaccard similarity fallback if model not injected
            similarities = []
            for i in range(len(sentences) - 1):
                set1 = set(sentences[i].lower().split())
                set2 = set(sentences[i + 1].lower().split())
                union = set1.union(set2)
                sim = len(set1.intersection(set2)) / len(union) if union else 1.0
                similarities.append(sim)

        chunks: List[ChunkMetadata] = []
        current_chunk_sentences: List[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold and len(" ".join(current_chunk_sentences).split()) >= 30:
                # Topic shift boundary detected
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{document_id}_sem_{len(chunks)}"
                chunks.append(ChunkMetadata(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    strategy="semantic",
                    position=len(chunks),
                    text=chunk_text,
                    source=source
                ))
                current_chunk_sentences = [sentences[i + 1]]
            else:
                current_chunk_sentences.append(sentences[i + 1])

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{document_id}_sem_{len(chunks)}"
            chunks.append(ChunkMetadata(
                document_id=document_id,
                chunk_id=chunk_id,
                strategy="semantic",
                position=len(chunks),
                text=chunk_text,
                source=source
            ))

        return chunks
