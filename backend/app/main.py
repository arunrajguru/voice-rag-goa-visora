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


def build_sample_fallback_index(
    dense_retriever: DenseRetriever,
    bm25_retriever: BM25Retriever,
):
    """
    Creates a tiny fallback knowledge base when the real index
    files are unavailable.
    """

    logger.info(
        "Index files not found. "
        "Creating lightweight sample knowledge base..."
    )

    sample_docs = [
        (
            "doc_1",
            "MSMARCO-XI is a multilingual search and passage "
            "retrieval benchmark dataset developed for Indian "
            "languages and English QA benchmarking.",
        ),
        (
            "doc_2",
            "Retrieval-Augmented Generation RAG combines "
            "retrieval with language models to ground answers "
            "using relevant source documents.",
        ),
        (
            "doc_3",
            "Sarvam AI provides speech-to-text models optimized "
            "for Indian languages, accents and multilingual "
            "voice recognition.",
        ),
        (
            "doc_4",
            "Adaptive chunking selects sentence-aware or semantic "
            "chunking strategies depending on the type of query.",
        ),
        (
            "doc_5",
            "Grounding guardrails verify that generated answers "
            "are supported by retrieved source context.",
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
                source="MSMARCO-XI",
            )
        )

    dense_retriever.build_index(chunks)
    bm25_retriever.build_index(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Initializing Voice RAG Backend Services..."
    )

    # ---------------------------------------------------------
    # Lightweight Dense Retriever
    # ---------------------------------------------------------

    dense_retriever = DenseRetriever(
        settings.EMBEDDING_MODEL
    )

    # ---------------------------------------------------------
    # BM25 Retriever
    # ---------------------------------------------------------

    bm25_retriever = BM25Retriever()

    # ---------------------------------------------------------
    # Try loading existing metadata + BM25 index.
    #
    # We intentionally DO NOT load the old FAISS index.
    # The lightweight DenseRetriever rebuilds its own index
    # from the metadata text.
    # ---------------------------------------------------------

    indexes_loaded = False

    try:

        metadata_exists = os.path.exists(
            settings.METADATA_PATH
        )

        bm25_exists = os.path.exists(
            settings.BM25_PATH
        )

        if metadata_exists and bm25_exists:

            logger.info(
                "Loading metadata and BM25 index..."
            )

            with open(
                settings.METADATA_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(file)

            dense_retriever.load_index(
                settings.INDEX_PATH,
                metadata,
            )

            bm25_retriever.load_index(
                settings.BM25_PATH
            )

            logger.info(
                f"Successfully loaded "
                f"{len(metadata)} document chunks."
            )

            indexes_loaded = True

    except Exception as error:

        logger.warning(
            f"Could not load existing indexes: {error}"
        )

    # ---------------------------------------------------------
    # Fallback index
    # ---------------------------------------------------------

    if not indexes_loaded:

        logger.info(
            "Using lightweight fallback knowledge base."
        )

        build_sample_fallback_index(
            dense_retriever,
            bm25_retriever,
        )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Do NOT warm up SentenceTransformer here.
    #
    # The previous implementation loaded a large ML model
    # and performed an embedding warmup.
    # That is removed for Render's 512 MB environment.
    # ---------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        alpha=settings.DENSE_WEIGHT,
    )

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


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
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

    port = int(
        os.environ.get(
            "PORT",
            getattr(settings, "PORT", 10000),
        )
    )

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
    )
