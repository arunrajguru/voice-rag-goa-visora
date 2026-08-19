from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

@dataclass
class ChunkMetadata:
    document_id: str
    chunk_id: str
    strategy: str  # fixed, sentence, semantic
    position: int
    text: str
    source: str = "MSMARCO-XI"
    score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0

@dataclass
class StageTimings:
    stt_ms: float = 0.0
    preprocessing_ms: float = 0.0
    embedding_ms: float = 0.0
    faiss_ms: float = 0.0
    bm25_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "stt_ms": round(self.stt_ms, 2),
            "preprocessing_ms": round(self.preprocessing_ms, 2),
            "embedding_ms": round(self.embedding_ms, 2),
            "faiss_ms": round(self.faiss_ms, 2),
            "bm25_ms": round(self.bm25_ms, 2),
            "reranking_ms": round(self.reranking_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "grounding_ms": round(self.grounding_ms, 2),
            "total_ms": round(self.total_ms, 2)
        }

@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    refusal_message: Optional[str] = None

class RAGRequest(BaseModel):
    query: str = Field(..., description="User query text")
    top_k: Optional[int] = Field(default=None, description="Number of candidates to retrieve")

class VoiceRAGResponse(BaseModel):
    transcript: str
    answer: str
    grounded: bool
    refused: bool
    confidence: float
    query_category: str
    chunk_strategy: str
    sources: List[Dict[str, Any]]
    timings: Dict[str, float]
    request_id: str
