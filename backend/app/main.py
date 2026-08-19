import os
import json
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.services.pipeline_harness import PipelineHarness
from app.models.data_models import ChunkMetadata
from app.utils.logger import logger

def build_sample_fallback_index(dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever):
    """Creates a lightweight sample dataset in memory if index files have not been generated yet."""
    logger.info("Index files not found. Creating temporary in-memory sample knowledge base for instant demo execution...")
    sample_docs = [
        ("doc_1", "MSMARCO-XI is a multilingual search and passage retrieval benchmark dataset developed for Indian languages and English QA benchmarking."),
        ("doc_2", "Retrieval-Augmented Generation (RAG) combines dense vector retrieval using FAISS with BM25 sparse lexical search to ground LLM answers."),
        ("doc_3", "Sarvam AI provides high-performance Speech-to-Text Saaras models optimized for Indian languages, accents, and multilingual voice recognition."),
        ("doc_4", "Adaptive chunking selects sentence-aware or semantic embedding breakpoint strategies depending on whether a query is factual or broad."),
        ("doc_5", "Grounding guardrails verify that generated answers contain verbatim facts supported by retrieved source context passages.")
    ]
    chunks = []
    for doc_id, text in sample_docs:
        chunks.append(ChunkMetadata(
            document_id=doc_id,
            chunk_id=f"{doc_id}_c0",
            strategy="sentence",
            position=0,
            text=text,
            source="MSMARCO-XI"
        ))
    dense_retriever.build_index(chunks)
    bm25_retriever.build_index(chunks)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Voice RAG Backend Services...")
    
    dense_retriever = DenseRetriever(settings.EMBEDDING_MODEL)
    bm25_retriever = BM25Retriever()
    
    # Try loading existing disk indexes, else load fallback in-memory index
    if os.path.exists(settings.INDEX_PATH) and os.path.exists(settings.METADATA_PATH) and os.path.exists(settings.BM25_PATH):
        try:
            logger.info("Loading FAISS vector index and BM25 sparse index into memory...")
            with open(settings.METADATA_PATH, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            dense_retriever.load_index(settings.INDEX_PATH, meta_json)
            bm25_retriever.load_index(settings.BM25_PATH)
            logger.info(f"Successfully loaded {len(meta_json)} document chunks into memory!")
        except Exception as e:
            logger.error(f"Error loading index files from disk: {e}")
            build_sample_fallback_index(dense_retriever, bm25_retriever)
    else:
        build_sample_fallback_index(dense_retriever, bm25_retriever)

    # Warmup embedding model
    logger.info("Warming up embedding model for zero-cold-start inference...")
    dense_retriever.model.encode(["Warmup query"], normalize_embeddings=True)

    hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever, alpha=settings.DENSE_WEIGHT)
    app.state.pipeline_harness = PipelineHarness(hybrid_retriever)

    logger.info("Voice RAG Backend Startup Complete! Ready to serve requests.")
    yield
    logger.info("Shutting down Voice RAG Backend Services...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
