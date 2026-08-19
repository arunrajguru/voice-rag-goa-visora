# Latency Optimization & Benchmark Methodology

## Architectural Latency Drivers (<200 ms target)
1. **In-Memory Indexes**: FAISS `IndexFlatIP` and BM25 indexes are kept in RAM.
2. **Startup Warmup**: SentenceTransformer and embedding models are loaded once during server lifespan startup. Zero runtime model cold starts.
3. **Stage Instrumentation**: Every stage timing (`stt_ms`, `preprocessing_ms`, `embedding_ms`, `faiss_ms`, `bm25_ms`, `reranking_ms`, `generation_ms`, `grounding_ms`, `total_ms`) is tracked via high-precision timers.

## Benchmark Metrics
The benchmark runner (`scripts/benchmark.py`) evaluates 100+ categorized queries and reports:
- P50 (Median)
- P70 (70th percentile)
- P100 (Maximum)
- Mean & Min/Max metrics
- Grounded answer rate, refusal rate, and retrieval success rate.
