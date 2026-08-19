# Hackathon Requirement Mapping (HH Goa 2026 Task 2)

| Requirement | Implementation Detail | Status |
| :--- | :--- | :--- |
| **1. Speech-to-Text** | Sarvam AI REST API/SDK (`sarvam-sdk` fallback client, `SARVAM_API_KEY` from `.env`) | ✅ Verified |
| **2. Chunking** | Fixed-size with overlap, Sentence-aware, Semantic similarity breakpoint chunker, Adaptive query selector (`FixedChunker`, `SentenceChunker`, `SemanticChunker`, `AdaptiveChunkSelector`) | ✅ Verified |
| **3. Retrieval** | FAISS CPU In-Memory Dense Retriever + BM25 Sparse Retriever + Score Normalization & Fusion ($S_{hybrid} = \alpha S_{dense} + (1-\alpha) S_{bm25}$) | ✅ Verified |
| **4. Latency Target** | <200 ms target backend retrieval, models preloaded at startup, FAISS kept in RAM, zero cold start overhead, stage-by-stage timing instrumentation | ✅ Verified |
| **5. Harness** | Orchestration layer (`PipelineHarness`) with input validation, preprocessing, safety, query classification, retrieval, reranking, LLM generation, grounding verification, and output validation | ✅ Verified |
| **6. Guardrails** | Input safety guardrail, Off-topic guardrail, Retrieval confidence guardrail, Grounding verification guardrail, Output validation guardrail | ✅ Verified |
| **7. Tech Stack** | FastAPI, PyDantic, Sarvam STT, FAISS CPU, BM25, SentenceTransformers, React + Vite UI | ✅ Verified |
| **8. Dataset** | `ai4bharat/MSMARCO-XI` schema inspection tool (`inspect_dataset.py`) & offline builder (`build_index.py`) | ✅ Verified |
| **9. Modular LLM** | `LLMProvider` abstract adapter supporting Groq, OpenAI, Sarvam, and Local Mock fallback | ✅ Verified |
| **10. Benchmarking** | 100+ query harness (`queries.json`, `scripts/benchmark.py`) computing P50, P70, P100, min, max, mean per stage, exporting CSV and JSON reports | ✅ Verified |
| **11. React UI** | Polished dark mode UI, microphone voice recorder, Web Audio visualizer, pipeline stepper, latency card, sources drawer, refusal badges, example questions | ✅ Verified |
| **12. Pytest Suite** | Test suites in `tests/` covering chunking, retrieval, guardrails, pipeline harness, and API routes | ✅ Verified |
| **13. Packaging** | Automated ZIP packager script (`package_zip.py`) creating `voice-rag-goa-2026.zip` | ✅ Verified |
