# Multi-Strategy & Adaptive Chunking

## 1. Fixed-size Chunking (`FixedChunker`)
- **Parameters**: `chunk_size=500` characters/words, `overlap=50`.
- **Rationale**: Uniform density and bounded memory footprint for fixed-embedding context windows.

## 2. Sentence-Aware Chunking (`SentenceChunker`)
- **Parameters**: `target_words=150`.
- **Rationale**: Splits text exclusively at sentence boundaries (`.!?`), preserving complete grammatical thoughts.

## 3. Semantic Breakpoint Chunking (`SemanticChunker`)
- **Parameters**: `similarity_threshold=0.5`.
- **Rationale**: Measures cosine similarity between consecutive sentence vectors. A topic breakpoint is created when adjacent similarity falls below threshold.

## 4. Adaptive Strategy Selection (`AdaptiveChunkSelector`)
Query classification dynamically maps queries to retrieval parameters:
- **exact**: Sentence chunking, $\alpha = 0.3$ (favors BM25).
- **factual**: Sentence chunking, $\alpha = 0.5$ (balanced).
- **semantic**: Semantic chunking, $\alpha = 0.7$ (favors dense FAISS).
- **broad**: Fixed chunking, $\alpha = 0.6$.
