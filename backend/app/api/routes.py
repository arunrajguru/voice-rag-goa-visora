import time

from typing import (
    Dict,
    Any,
    Optional
)

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Request
)

from app.models.data_models import (
    RAGRequest,
    VoiceRAGResponse
)

from app.api.schemas import (
    HealthResponse,
    STTResponse,
    ConfigResponse,
    MetricsResponse
)

from app.config import settings

from app.services.pipeline_harness import (
    PipelineHarness
)

from app.services.stt_service import (
    SarvamSTTService
)

from app.utils.logger import logger


router = APIRouter()


# ==========================================================
# GLOBAL METRICS
# ==========================================================

METRICS_DATA = {
    "total_queries": 0,
    "total_rag_ms": 0.0,
    "total_stt_ms": 0.0,
    "grounded_count": 0,
    "refusal_count": 0
}


# ==========================================================
# PIPELINE HARNESS DEPENDENCY
# ==========================================================

def get_harness(
    request: Request
) -> PipelineHarness:

    harness = getattr(
        request.app.state,
        "pipeline_harness",
        None
    )

    if not harness:

        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline harness index "
                "not loaded or uninitialized."
            )
        )

    return harness


# ==========================================================
# HEALTH
# ==========================================================

@router.get(
    "/health",
    response_model=HealthResponse
)
async def health_check(
    request: Request
):

    harness = getattr(
        request.app.state,
        "pipeline_harness",
        None
    )

    loaded = harness is not None

    total_docs = (
        len(
            harness
            .hybrid_retriever
            .dense
            .metadata_store
        )
        if loaded
        else 0
    )

    return HealthResponse(

        status=(
            "healthy"
            if loaded
            else "degraded"
        ),

        version=settings.VERSION,

        faiss_index_loaded=loaded,

        bm25_index_loaded=loaded,

        total_documents_indexed=total_docs
    )


# ==========================================================
# CONFIG
# ==========================================================

@router.get(
    "/api/config",
    response_model=ConfigResponse
)
async def get_config():

    return ConfigResponse(

        app_name=settings.APP_NAME,

        embedding_model=settings.EMBEDDING_MODEL,

        llm_provider=settings.LLM_PROVIDER,

        llm_model=settings.LLM_MODEL,

        top_k=settings.TOP_K,

        final_k=settings.FINAL_K,

        dense_weight_alpha=settings.DENSE_WEIGHT,

        similarity_threshold=settings.SIMILARITY_THRESHOLD,

        grounding_threshold=settings.GROUNDING_THRESHOLD
    )


# ==========================================================
# STT ONLY
# ==========================================================

@router.post(
    "/api/stt",
    response_model=STTResponse
)
async def speech_to_text(
    file: UploadFile = File(...)
):

    stt_service = SarvamSTTService()

    start = time.perf_counter()

    # ------------------------------------------------------
    # Read uploaded audio
    # ------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Empty audio file provided."
        )

    # ------------------------------------------------------
    # File size validation
    # ------------------------------------------------------

    if len(contents) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="Audio file exceeds 10MB limit."
        )

    filename = (
        file.filename
        or
        "recording.webm"
    )

    logger.info(
        f"STT request: "
        f"filename={filename}, "
        f"content_type={file.content_type}, "
        f"size={len(contents)} bytes"
    )

    # ------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------

    try:

        transcript = (
            await stt_service.transcribe_audio(
                audio_bytes=contents,
                filename=filename,
                language_code="unknown"
            )
        )

    except Exception as error:

        logger.exception(
            f"STT failed: {error}"
        )

        raise HTTPException(
            status_code=502,
            detail=str(error)
        )

    # ------------------------------------------------------
    # Timing
    # ------------------------------------------------------

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    logger.info(
        f"STT completed in "
        f"{elapsed_ms:.2f} ms: "
        f"{transcript}"
    )

    return STTResponse(
        transcript=transcript,
        stt_ms=round(
            elapsed_ms,
            2
        )
    )


# ==========================================================
# TEXT RAG
# ==========================================================

@router.post(
    "/api/rag",
    response_model=VoiceRAGResponse
)
async def text_rag(
    request_data: RAGRequest,
    harness: PipelineHarness = Depends(
        get_harness
    )
):

    response = (
        await harness.execute_voice_pipeline(
            audio_bytes=None,
            query_text=request_data.query
        )
    )

    METRICS_DATA[
        "total_queries"
    ] += 1

    METRICS_DATA[
        "total_rag_ms"
    ] += response.timings[
        "total_ms"
    ]

    if response.grounded:

        METRICS_DATA[
            "grounded_count"
        ] += 1

    if response.refused:

        METRICS_DATA[
            "refusal_count"
        ] += 1

    return response


# ==========================================================
# VOICE RAG
# ==========================================================

@router.post(
    "/api/voice-rag",
    response_model=VoiceRAGResponse
)
async def voice_rag(
    file: UploadFile = File(...),
    harness: PipelineHarness = Depends(
        get_harness
    )
):

    # ------------------------------------------------------
    # Read audio
    # ------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty."
        )

    # ------------------------------------------------------
    # Size validation
    # ------------------------------------------------------

    if len(contents) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="Audio file exceeds 10MB limit."
        )

    # ------------------------------------------------------
    # Preserve REAL filename
    # ------------------------------------------------------

    filename = (
        file.filename
        or
        "recording.webm"
    )

    logger.info(
        f"Voice RAG request: "
        f"filename={filename}, "
        f"content_type={file.content_type}, "
        f"size={len(contents)} bytes"
    )

    # ------------------------------------------------------
    # Execute pipeline
    # ------------------------------------------------------

    try:

        response = (
            await harness.execute_voice_pipeline(
                audio_bytes=contents,
                audio_filename=filename
            )
        )

    except Exception as error:

        logger.exception(
            f"Voice RAG pipeline failed: {error}"
        )

        raise HTTPException(
            status_code=502,
            detail=str(error)
        )

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    METRICS_DATA[
        "total_queries"
    ] += 1

    METRICS_DATA[
        "total_rag_ms"
    ] += response.timings.get(
        "total_ms",
        0.0
    )

    METRICS_DATA[
        "total_stt_ms"
    ] += response.timings.get(
        "stt_ms",
        0.0
    )

    if response.grounded:

        METRICS_DATA[
            "grounded_count"
        ] += 1

    if response.refused:

        METRICS_DATA[
            "refusal_count"
        ] += 1

    return response


# ==========================================================
# METRICS
# ==========================================================

@router.get(
    "/api/metrics",
    response_model=MetricsResponse
)
async def get_metrics():

    total = max(
        1,
        METRICS_DATA[
            "total_queries"
        ]
    )

    return MetricsResponse(

        total_queries_processed=(
            METRICS_DATA[
                "total_queries"
            ]
        ),

        average_rag_latency_ms=round(
            METRICS_DATA[
                "total_rag_ms"
            ] / total,
            2
        ),

        average_stt_latency_ms=round(
            METRICS_DATA[
                "total_stt_ms"
            ] / total,
            2
        ),

        grounded_rate=round(
            METRICS_DATA[
                "grounded_count"
            ] / total,
            2
        ),

        refusal_rate=round(
            METRICS_DATA[
                "refusal_count"
            ] / total,
            2
        )
    )
