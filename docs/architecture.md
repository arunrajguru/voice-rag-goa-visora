# System Architecture & Pipeline Harness

## Pipeline Flow

```mermaid
graph TD
    A[Voice Input / Audio Blob] -->|HTTP POST /api/voice-rag| B[Sarvam Speech-to-Text]
    B -->|Transcript Text| C[Input Preprocessing & Cleaning]
    C --> D[Input Safety Guardrail]
    D -->|Passed| E[Off-Topic Guardrail]
    E -->|Passed| F[Query Classifier & Adaptive Selector]
    F --> G[Hybrid Retrieval Engine]
    G -->|Dense Vector Search| H[FAISS In-Memory Index]
    G -->|Sparse Lexical Search| I[BM25 Index]
    H & I --> J[Min-Max Score Normalization & Fusion]
    J --> K[Reranker & Deduplication]
    K --> L[Retrieval Confidence Guardrail]
    L -->|Passed Contexts| M[LLM Adapter - Groq / OpenAI / Sarvam / Mock]
    M --> N[Grounding Verification Guardrail]
    N --> O[Output Validation Guardrail]
    O --> P[Structured JSON Response]
```

## Harness Orchestration & Error Isolation
Every stage in `PipelineHarness` captures microsecond timing (`time.perf_counter()`), catches exceptions, records audit metadata, and returns a predictable structured response payload.
