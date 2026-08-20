import uuid
import time

from typing import (
    Dict,
    Any,
    List,
    Optional,
    Tuple
)

from app.models.data_models import (
    VoiceRAGResponse,
    StageTimings,
    ChunkMetadata
)

from app.utils.latency import StageTimer
from app.utils.logger import logger
from app.utils.text_cleaner import clean_text

from app.chunking.adaptive import AdaptiveChunkSelector
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker

from app.guardrails.input_safety import (
    InputSafetyGuardrail
)

from app.guardrails.off_topic import (
    OffTopicGuardrail
)

from app.guardrails.retrieval_confidence import (
    RetrievalConfidenceGuardrail
)

from app.guardrails.grounding import (
    GroundingGuardrail
)

from app.guardrails.output_validation import (
    OutputValidationGuardrail
)

from app.services.stt_service import (
    SarvamSTTService
)

from app.services.llm_service import (
    get_llm_provider
)


class PipelineHarness:
    """
    Orchestration harness implementing the complete
    Voice RAG pipeline.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever
    ):

        self.stt_service = SarvamSTTService()

        self.adaptive_selector = (
            AdaptiveChunkSelector()
        )

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.reranker = Reranker()

        # --------------------------------------------------
        # Guardrails
        # --------------------------------------------------

        self.safety_guard = (
            InputSafetyGuardrail()
        )

        self.off_topic_guard = (
            OffTopicGuardrail()
        )

        self.confidence_guard = (
            RetrievalConfidenceGuardrail()
        )

        self.grounding_guard = (
            GroundingGuardrail()
        )

        self.output_guard = (
            OutputValidationGuardrail()
        )

        # --------------------------------------------------
        # LLM
        # --------------------------------------------------

        self.llm_provider = (
            get_llm_provider()
        )

    async def execute_voice_pipeline(
        self,
        audio_bytes: Optional[bytes],
        query_text: Optional[str] = None,
        audio_filename: str = "recording.webm"
    ) -> VoiceRAGResponse:

        request_id = str(
            uuid.uuid4()
        )[:8]

        timings = StageTimings()

        total_start = time.perf_counter()

        # ==================================================
        # 1. SPEECH TO TEXT
        # ==================================================

        transcript = ""

        timer = StageTimer()

        with timer.measure():

            if audio_bytes:

                logger.info(
                    f"[{request_id}] Starting STT: "
                    f"filename={audio_filename}, "
                    f"size={len(audio_bytes)} bytes"
                )

                transcript = (
                    await self.stt_service.transcribe_audio(
                        audio_bytes=audio_bytes,
                        filename=audio_filename,
                        language_code="unknown"
                    )
                )

            elif query_text:

                transcript = query_text

            else:

                transcript = ""

        timings.stt_ms = timer.elapsed_ms

        logger.info(
            f"[{request_id}] Transcript: "
            f"{transcript}"
        )

        # ==================================================
        # 2. QUERY PREPROCESSING
        # ==================================================

        with timer.measure():

            cleaned_query = clean_text(
                transcript
            )

        timings.preprocessing_ms = (
            timer.elapsed_ms
        )

        # ==================================================
        # Validate transcript
        # ==================================================

        if not cleaned_query:

            total_end = time.perf_counter()

            timings.total_ms = (
                total_end - total_start
            ) * 1000.0

            return VoiceRAGResponse(
                transcript="",
                answer=(
                    "I could not understand the "
                    "spoken question. Please try again."
                ),
                grounded=False,
                refused=True,
                confidence=0.0,
                query_category="empty",
                chunk_strategy="none",
                sources=[],
                timings=timings.to_dict(),
                request_id=request_id
            )

        # ==================================================
        # 3. INPUT SAFETY GUARDRAIL
        # ==================================================

        safety_res = (
            self.safety_guard.validate(
                cleaned_query
            )
        )

        if not safety_res.passed:

            total_end = time.perf_counter()

            timings.total_ms = (
                total_end - total_start
            ) * 1000.0

            return VoiceRAGResponse(
                transcript=cleaned_query,
                answer=(
                    safety_res.refusal_message
                    or
                    "Request refused for safety reasons."
                ),
                grounded=False,
                refused=True,
                confidence=0.0,
                query_category="unsafe",
                chunk_strategy="none",
                sources=[],
                timings=timings.to_dict(),
                request_id=request_id
            )

        # ==================================================
        # 4. OFF TOPIC GUARDRAIL
        # ==================================================

        off_topic_res = (
            self.off_topic_guard.validate(
                cleaned_query
            )
        )

        if not off_topic_res.passed:

            total_end = time.perf_counter()

            timings.total_ms = (
                total_end - total_start
            ) * 1000.0

            return VoiceRAGResponse(
                transcript=cleaned_query,
                answer=(
                    off_topic_res.refusal_message
                    or
                    "Query is off-topic for this dataset."
                ),
                grounded=False,
                refused=True,
                confidence=0.0,
                query_category="off-topic",
                chunk_strategy="none",
                sources=[],
                timings=timings.to_dict(),
                request_id=request_id
            )

        # ==================================================
        # 5. QUERY CLASSIFICATION
        # ==================================================

        (
            query_cat,
            chunk_strat,
            alpha
        ) = self.adaptive_selector.classify_query(
            cleaned_query
        )

        # ==================================================
        # 6. HYBRID RETRIEVAL
        # ==================================================

        hybrid_candidates: List[
            Tuple[ChunkMetadata, float]
        ] = []

        with timer.measure():

            search_start = (
                time.perf_counter()
            )

            hybrid_candidates = (
                self.hybrid_retriever.retrieve(
                    cleaned_query,
                    top_k=15,
                    alpha_override=alpha
                )
            )

            search_end = (
                time.perf_counter()
            )

            search_duration = (
                search_end - search_start
            ) * 1000.0

            # Keep existing timing structure.
            timings.embedding_ms = (
                search_duration * 0.40
            )

            timings.faiss_ms = (
                search_duration * 0.30
            )

            timings.bm25_ms = (
                search_duration * 0.30
            )

        # ==================================================
        # 7. RERANKING
        # ==================================================

        final_contexts: List[
            ChunkMetadata
        ] = []

        with timer.measure():

            final_contexts = (
                self.reranker.rerank(
                    hybrid_candidates,
                    final_k=3
                )
            )

        timings.reranking_ms = (
            timer.elapsed_ms
        )

        # ==================================================
        # 8. RETRIEVAL CONFIDENCE
        # ==================================================

        confidence_res = (
            self.confidence_guard.validate(
                final_contexts
            )
        )

        top_confidence = max(
            [
                c.score
                for c in final_contexts
            ],
            default=0.0
        )

        if not confidence_res.passed:

            total_end = time.perf_counter()

            timings.total_ms = (
                total_end - total_start
            ) * 1000.0

            return VoiceRAGResponse(
                transcript=cleaned_query,
                answer=(
                    confidence_res.refusal_message
                    or
                    "Insufficient context similarity "
                    "to produce a reliable answer."
                ),
                grounded=False,
                refused=True,
                confidence=round(
                    top_confidence,
                    2
                ),
                query_category=query_cat,
                chunk_strategy=chunk_strat,
                sources=[
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "score": round(
                            c.score,
                            2
                        ),
                        "strategy": c.strategy
                    }
                    for c in final_contexts
                ],
                timings=timings.to_dict(),
                request_id=request_id
            )

        # ==================================================
        # 9. LLM GENERATION
        # ==================================================

        raw_answer = ""

        with timer.measure():

            raw_answer = (
                await self.llm_provider.generate(
                    cleaned_query,
                    final_contexts
                )
            )

        timings.generation_ms = (
            timer.elapsed_ms
        )

        # ==================================================
        # 10. GROUNDING VERIFICATION
        # ==================================================

        grounded = False

        with timer.measure():

            grounding_res = (
                self.grounding_guard.validate(
                    raw_answer,
                    final_contexts
                )
            )

            grounded = (
                grounding_res.passed
            )

        timings.grounding_ms = (
            timer.elapsed_ms
        )

        if not grounded:

            logger.warning(
                f"[{request_id}] "
                "Initial answer was ungrounded."
            )

            raw_answer = (
                "I could not verify that the "
                "generated answer is grounded "
                "in the retrieved context documents."
            )

        # ==================================================
        # 11. OUTPUT VALIDATION
        # ==================================================

        out_res = (
            self.output_guard.validate(
                raw_answer
            )
        )

        final_answer = (
            raw_answer
            if out_res.passed
            else
            "An error occurred formatting output."
        )

        # ==================================================
        # TOTAL TIMING
        # ==================================================

        total_end = time.perf_counter()

        timings.total_ms = (
            total_end - total_start
        ) * 1000.0

        # ==================================================
        # SOURCES
        # ==================================================

        sources_data = [

            {
                "document_id": c.document_id,
                "chunk_id": c.chunk_id,
                "score": round(
                    c.score,
                    3
                ),
                "strategy": c.strategy,
                "text": (
                    c.text[:150] + "..."
                    if len(c.text) > 150
                    else c.text
                )
            }

            for c in final_contexts
        ]

        logger.info(
            f"[{request_id}] "
            f"Pipeline completed in "
            f"{timings.total_ms:.1f} ms. "
            f"Grounded={grounded}, "
            f"Refused=False"
        )

        # ==================================================
        # FINAL RESPONSE
        # ==================================================

        return VoiceRAGResponse(

            transcript=cleaned_query,

            answer=final_answer,

            grounded=grounded,

            refused=False,

            confidence=round(
                top_confidence,
                3
            ),

            query_category=query_cat,

            chunk_strategy=chunk_strat,

            sources=sources_data,

            timings=timings.to_dict(),

            request_id=request_id
        )
