import re
from typing import List
from app.models.data_models import ChunkMetadata
from app.utils.text_cleaner import clean_text

class SentenceChunker:
    """Sentence boundary-aware chunking.
    
    Why this exists:
    Prevents slicing sentences mid-thought, preserving grammatical integrity and logical context
    for factual lookups and precise answers.
    """
    def __init__(self, target_words: int = 150):
        self.target_words = target_words

    def _split_sentences(self, text: str) -> List[str]:
        # Split on punctuation (.!?) followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, document_id: str, text: str, source: str = "MSMARCO-XI") -> List[ChunkMetadata]:
        cleaned = clean_text(text)
        if not cleaned:
            return []
        
        sentences = self._split_sentences(cleaned)
        chunks: List[ChunkMetadata] = []
        current_sentences: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > self.target_words and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunk_id = f"{document_id}_sent_{len(chunks)}"
                chunks.append(ChunkMetadata(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    strategy="sentence",
                    position=len(chunks),
                    text=chunk_text,
                    source=source
                ))
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sentence_words

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = f"{document_id}_sent_{len(chunks)}"
            chunks.append(ChunkMetadata(
                document_id=document_id,
                chunk_id=chunk_id,
                strategy="sentence",
                position=len(chunks),
                text=chunk_text,
                source=source
            ))

        return chunks
