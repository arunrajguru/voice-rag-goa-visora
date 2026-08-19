# FAISS + BM25 Hybrid Retrieval & Reranking

## Score Normalization & Weighted Fusion
Dense scores ($S_{dense}$) from FAISS vector search and sparse scores ($S_{bm25}$) from BM25 are normalized using min-max scaling to $[0, 1]$ range:

$$S_{norm} = \frac{S - S_{min}}{S_{max} - S_{min}}$$

The final hybrid rank score is computed as:

$$S_{hybrid} = \alpha \cdot S_{dense, norm} + (1 - \alpha) \cdot S_{bm25, norm}$$

where $\alpha \in [0, 1]$ is dynamically tuned per query category.

## Lightweight Reranking & Deduplication
To maintain the <200 ms backend retrieval latency budget, lightweight text-overlap deduplication and length-normalization factors are applied instead of heavy cross-encoders.
