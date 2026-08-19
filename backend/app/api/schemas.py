from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    version: str
    faiss_index_loaded: bool
    bm25_index_loaded: bool
    total_documents_indexed: int

class STTResponse(BaseModel):
    transcript: str
    stt_ms: float

class ConfigResponse(BaseModel):
    app_name: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    top_k: int
    final_k: int
    dense_weight_alpha: float
    similarity_threshold: float
    grounding_threshold: float

class MetricsResponse(BaseModel):
    total_queries_processed: int
    average_rag_latency_ms: float
    average_stt_latency_ms: float
    grounded_rate: float
    refusal_rate: float
