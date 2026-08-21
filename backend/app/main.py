import os
import json
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


# =========================================================
# LIGHTWEIGHT FALLBACK KNOWLEDGE BASE
# =========================================================

def build_fallback_index(
    dense_retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
):
    """
    Creates a small English knowledge base when the
    MSMARCO-XI indexes are unavailable.

    This keeps the Render deployment lightweight and
    avoids building the large MSMARCO-XI index at startup.
    """

    logger.info(
        "Creating lightweight fallback knowledge base..."
    )

    sample_docs = [
        (
            "doc_1",
            "MSMARCO-XI is a multilingual search and passage "
            "retrieval benchmark dataset developed for Indian "
            "languages and English question answering and "
            "passage retrieval research."
        ),

        (
            "doc_2",
            "Retrieval-Augmented Generation, commonly called RAG, "
            "combines information retrieval with a language model. "
            "The retriever finds relevant passages and the language "
            "model uses those passages to generate a grounded answer."
        ),

        (
            "doc_3",
            "Sarvam AI develops artificial intelligence models and "
            "applications focused on Indian languages and users. "
            "Sarvam AI provides speech and language technologies "
            "including speech recognition and text generation."
        ),

        (
            "doc_4",
            "Sarvam AI's Saaras family refers to speech recognition "
            "technology designed for Indian languages and multilingual "
            "speech-to-text applications."
        ),

        (
            "doc_5",
            "Adaptive chunking selects an appropriate text chunking "
            "strategy based on the type of query. Sentence-aware "
            "chunking and semantic chunking can preserve useful context."
        ),

        (
            "doc_6",
            "Hybrid retrieval combines dense vector retrieval and "
            "BM25 keyword retrieval. Dense retrieval captures semantic "
            "similarity while BM25 is effective for exact keywords."
        ),

        (
            "doc_7",
            "A grounding guardrail checks whether a generated answer "
            "is supported by the retrieved context. If the answer "
            "cannot be supported, the system can return Not Found."
        ),

        (
            "doc_8",
            "A reranker takes retrieved candidates and ranks them "
            "according to their relevance to the user's question. "
            "The highest quality passages are then supplied to the LLM."
        ),

        (
            "doc_9",
            "FAISS is a library for efficient similarity search over "
            "dense vectors. It is commonly used in retrieval systems "
            "to find semantically similar documents."
        ),

        (
            "doc_10",
            "BM25 is a traditional information retrieval ranking "
            "algorithm that scores documents according to the "
            "occurrence and importance of query terms."
        ),
    ]

    chunks = []

    for doc_id, text in sample_docs:

        chunks.append(
            ChunkMetadata(
                document_id=doc_id,
                chunk_id=f"{doc_id}_c0",
                strategy="sentence",
                position=0,
                text=text,
                source="Voice RAG Knowledge Base",
            )
        )

    # Build both retrieval indexes in memory.

    dense_retriever.build_index(
        chunks
    )

    bm25_retriever.build_index(
        chunks
    )

    logger.info(
        f"Fallback knowledge base ready: "
        f"{len(chunks)} documents"
    )


# =========================================================
# REAL INDEX LOADER
# =========================================================

def load_real_indexes(
    dense_retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
) -> bool:
    """
    Attempts to load the real MSMARCO-XI indexes.

    Returns False if the indexes are unavailable.
    """

    metadata_path = settings.METADATA_PATH
    bm25_path = settings.BM25_PATH
    dense_path = settings.INDEX_PATH

    logger.info(
        "Checking MSMARCO-XI index files..."
    )

    metadata_exists = os.path.exists(
        metadata_path
    )

    bm25_exists = os.path.exists(
        bm25_path
    )

    dense_exists = os.path.exists(
        dense_path
    )

    logger.info(
        f"Metadata: {metadata_path} "
        f"exists={metadata_exists}"
    )

    logger.info(
        f"BM25: {bm25_path} "
        f"exists={bm25_exists}"
    )

    logger.info(
        f"Dense: {dense_path} "
        f"exists={dense_exists}"
    )

    if not metadata_exists:

        logger.warning(
            "Metadata file not found."
        )

        return False

    if not bm25_exists:

        logger.warning(
            "BM25 index not found."
        )

        return False

    if not dense_exists:

        logger.warning(
            "Dense FAISS index not found."
        )

        return False

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        if not metadata:

            logger.warning(
                "Metadata file is empty."
            )

            return False

        logger.info(
            f"Loading {len(metadata)} chunks..."
        )

        dense_retriever.load_index(
            dense_path,
            metadata,
        )

        bm25_retriever.load_index(
            bm25_path
        )

        logger.info(
            "=============================================="
        )

        logger.info(
            "REAL MSMARCO-XI INDEX LOADED SUCCESSFULLY"
        )

        logger.info(
            f"Chunks loaded: {len(metadata)}"
        )

        logger.info(
            "=============================================="
        )

        return True

    except Exception as error:

        logger.exception(
            f"Failed to load real indexes: {error}"
        )

        return False


# =========================================================
# FASTAPI LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Initializing Voice RAG Backend Services..."
    )

    # -----------------------------------------------------
    # Dense Retriever
    # -----------------------------------------------------

    dense_retriever = DenseRetriever(
        settings.EMBEDDING_MODEL
    )

    # -----------------------------------------------------
    # BM25 Retriever
    # -----------------------------------------------------

    bm25_retriever = BM25Retriever()

    # -----------------------------------------------------
    # Try real MSMARCO-XI indexes
    # -----------------------------------------------------

    indexes_loaded = load_real_indexes(
        dense_retriever,
        bm25_retriever,
    )

    # -----------------------------------------------------
    # Use fallback if real indexes are unavailable
    # -----------------------------------------------------

    if not indexes_loaded:

        logger.warning(
            "MSMARCO-XI indexes are unavailable."
        )

        logger.info(
            "Using lightweight fallback knowledge base."
        )

        build_fallback_index(
            dense_retriever,
            bm25_retriever,
        )

    # -----------------------------------------------------
    # Hybrid Retriever
    # -----------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        alpha=settings.DENSE_WEIGHT,
    )

    # -----------------------------------------------------
    # Pipeline Harness
    # -----------------------------------------------------

    app.state.pipeline_harness = PipelineHarness(
        hybrid_retriever
    )

    logger.info(
        "Voice RAG Backend Startup Complete!"
    )

    yield

    logger.info(
        "Shutting down Voice RAG Backend Services..."
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "https://voice-rag-goa-visora.vercel.app",
        "https://voice-rag-goa-visora-gozuvv1pk-visora4.vercel.app",
    ],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
        "PATCH",
    ],

    allow_headers=[
        "*"
    ],
)


# =========================================================
# API ROUTES
# =========================================================

app.include_router(
    router
)


# =========================================================
# STARTUP
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            getattr(
                settings,
                "PORT",
                10000,
            ),
        )
    )

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
    )
