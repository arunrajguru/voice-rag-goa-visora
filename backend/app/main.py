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

from app.utils.logger import logger


def load_real_indexes(
    dense_retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
) -> bool:
    """
    Load the real MSMARCO-XI indexes.
    """

    metadata_path = settings.METADATA_PATH
    bm25_path = settings.BM25_PATH
    dense_path = settings.INDEX_PATH

    logger.info("Checking MSMARCO-XI index files...")

    logger.info(
        f"Metadata: {metadata_path} "
        f"exists={os.path.exists(metadata_path)}"
    )

    logger.info(
        f"BM25: {bm25_path} "
        f"exists={os.path.exists(bm25_path)}"
    )

    logger.info(
        f"Dense: {dense_path} "
        f"exists={os.path.exists(dense_path)}"
    )

    if not os.path.exists(metadata_path):
        logger.warning("Metadata file not found.")
        return False

    if not os.path.exists(bm25_path):
        logger.warning("BM25 index not found.")
        return False

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if not metadata:
            logger.warning("Metadata file is empty.")
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
            f"Failed to load indexes: {error}"
        )

        return False


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Initializing Voice RAG Backend Services..."
    )

    # =====================================================
    # DENSE RETRIEVER
    # =====================================================

    dense_retriever = DenseRetriever(
        settings.EMBEDDING_MODEL
    )

    # =====================================================
    # BM25 RETRIEVER
    # =====================================================

    bm25_retriever = BM25Retriever()

    # =====================================================
    # LOAD REAL INDEX
    # =====================================================

    indexes_loaded = load_real_indexes(
        dense_retriever,
        bm25_retriever,
    )

    # =====================================================
    # FALLBACK
    # =====================================================
    #
    # IMPORTANT:
    # We do NOT silently create fake knowledge.
    #
    # If the real index is missing, fail clearly.
    # This prevents the application from pretending
    # to answer using MSMARCO-XI.
    # =====================================================

    if not indexes_loaded:

        logger.error(
            "================================================"
        )

        logger.error(
            "MSMARCO-XI INDEXES ARE NOT AVAILABLE."
        )

        logger.error(
            "Run build_index.py before starting the backend."
        )

        logger.error(
            "================================================"
        )

        raise RuntimeError(
            "MSMARCO-XI indexes are missing. "
            "Run the indexing step before starting FastAPI."
        )

    # =====================================================
    # HYBRID RETRIEVER
    # =====================================================

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        alpha=settings.DENSE_WEIGHT,
    )

    # =====================================================
    # PIPELINE
    # =====================================================

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

    allow_headers=["*"],
)


# =========================================================
# API ROUTES
# =========================================================

app.include_router(router)


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
