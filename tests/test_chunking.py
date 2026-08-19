import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.chunking.fixed import FixedChunker
from app.chunking.sentence import SentenceChunker
from app.chunking.semantic import SemanticChunker
from app.chunking.adaptive import AdaptiveChunkSelector

def test_fixed_chunker_overlap():
    chunker = FixedChunker(chunk_size=10, overlap=3)
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    chunks = chunker.chunk_text("doc1", text)
    assert len(chunks) > 1
    assert chunks[0].strategy == "fixed"

def test_sentence_chunker():
    chunker = SentenceChunker(target_words=20)
    text = "First sentence here. Second sentence follows immediately. Third sentence ends the document."
    chunks = chunker.chunk_text("doc2", text)
    assert len(chunks) >= 1
    assert "First sentence" in chunks[0].text

def test_semantic_chunker():
    chunker = SemanticChunker(similarity_threshold=0.5)
    text = "Deep learning uses neural networks. Artificial intelligence is evolving fast. Cooking pasta requires boiling water."
    chunks = chunker.chunk_text("doc3", text)
    assert len(chunks) >= 1

def test_adaptive_chunk_selector():
    selector = AdaptiveChunkSelector()
    cat, strat, alpha = selector.classify_query("What is the capital of India?")
    assert cat == "factual"
    assert strat == "sentence"
