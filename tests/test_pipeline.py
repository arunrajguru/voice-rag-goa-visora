import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.models.data_models import ChunkMetadata
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.services.pipeline_harness import PipelineHarness

def test_pipeline_harness_execution():
    async def _test():
        sample_chunks = [
            ChunkMetadata("d1", "d1_c0", "sentence", 0, "MSMARCO-XI is a passage retrieval dataset developed by AI4Bharat.")
        ]
        dense = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2")
        dense.build_index(sample_chunks)
        sparse = BM25Retriever()
        sparse.build_index(sample_chunks)
        hybrid = HybridRetriever(dense, sparse)

        harness = PipelineHarness(hybrid)
        response = await harness.execute_voice_pipeline(audio_bytes=None, query_text="What is MSMARCO-XI?")
        
        assert response.transcript == "What is MSMARCO-XI?"
        assert response.refused is False
        assert response.timings["total_ms"] > 0
        assert len(response.sources) > 0

    asyncio.run(_test())
