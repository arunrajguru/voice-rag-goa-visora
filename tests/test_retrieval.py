import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.models.data_models import ChunkMetadata
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker

@pytest.fixture
def sample_chunks():
    return [
        ChunkMetadata("d1", "d1_c0", "sentence", 0, "MSMARCO-XI is a passage retrieval benchmark for AI QA systems."),
        ChunkMetadata("d2", "d2_c0", "sentence", 0, "Sarvam AI Saaras model provides speech to text for Indian languages."),
        ChunkMetadata("d3", "d3_c0", "sentence", 0, "FAISS vector retrieval enables fast similarity search in low latency.")
    ]

def test_dense_retriever(sample_chunks):
    retriever = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2")
    retriever.build_index(sample_chunks)
    results = retriever.retrieve("FAISS similarity search", top_k=2)
    assert len(results) == 2
    assert results[0][0].document_id == "d3"

def test_bm25_retriever(sample_chunks):
    retriever = BM25Retriever()
    retriever.build_index(sample_chunks)
    results = retriever.retrieve("Sarvam speech", top_k=2)
    assert len(results) >= 1
    assert results[0][0].document_id == "d2"

def test_hybrid_retriever(sample_chunks):
    dense = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2")
    dense.build_index(sample_chunks)
    sparse = BM25Retriever()
    sparse.build_index(sample_chunks)

    hybrid = HybridRetriever(dense, sparse, alpha=0.6)
    results = hybrid.retrieve("MSMARCO passage retrieval", top_k=2)
    assert len(results) >= 1
    assert results[0][0].document_id == "d1"

def test_reranker(sample_chunks):
    reranker = Reranker()
    candidates = [(c, 0.8) for c in sample_chunks]
    reranked = reranker.rerank(candidates, final_k=2)
    assert len(reranked) == 2
