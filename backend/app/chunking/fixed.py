from typing import List
from app.models.data_models import ChunkMetadata
from app.utils.text_cleaner import clean_text

class FixedChunker:
    """Fixed-size character/word chunking with configurable overlap.
    
    Why this exists:
    Ideal for uniform indexing density and deterministic memory footprint when chunk sizes
    need to match strict vector embedding window limits.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, document_id: str, text: str, source: str = "MSMARCO-XI") -> List[ChunkMetadata]:
        cleaned = clean_text(text)
        if not cleaned:
            return []
        
        words = cleaned.split()
        chunks: List[ChunkMetadata] = []
        step = max(1, self.chunk_size - self.overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            if not chunk_text.strip():
                continue
            chunk_id = f"{document_id}_fixed_{len(chunks)}"
            chunks.append(ChunkMetadata(
                document_id=document_id,
                chunk_id=chunk_id,
                strategy="fixed",
                position=len(chunks),
                text=chunk_text,
                source=source
            ))
            if i + self.chunk_size >= len(words):
                break
                
        return chunks
