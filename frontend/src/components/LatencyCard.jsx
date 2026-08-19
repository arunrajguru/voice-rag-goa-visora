import React from 'react';
import { Clock } from 'lucide-react';

export default function LatencyCard({ timings }) {
  if (!timings) return null;

  return (
    <div className="glass-panel" style={{ marginTop: '1rem' }}>
      <div className="card-header">
        <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Clock size={18} color="var(--accent-cyan)" /> Stage Latency Analytics Breakdown
        </h4>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Target: &lt;200 ms backend retrieval
        </span>
      </div>
      <div className="latency-grid">
        <div className="metric-pill">
          <label>STT Latency</label>
          <span>{timings.stt_ms?.toFixed(1) || 0} ms</span>
        </div>
        <div className="metric-pill">
          <label>Query Preprocess</label>
          <span>{timings.preprocessing_ms?.toFixed(1) || 0} ms</span>
        </div>
        <div className="metric-pill">
          <label>FAISS Search</label>
          <span>{timings.faiss_ms?.toFixed(1) || 0} ms</span>
        </div>
        <div className="metric-pill">
          <label>BM25 Search</label>
          <span>{timings.bm25_ms?.toFixed(1) || 0} ms</span>
        </div>
        <div className="metric-pill">
          <label>Reranking</label>
          <span>{timings.reranking_ms?.toFixed(1) || 0} ms</span>
        </div>
        <div className="metric-pill">
          <label>LLM Generation</label>
          <span>{timings.generation_ms?.toFixed(1) || 0} ms</span>
        </div>
        <div className="metric-pill">
          <label>Grounding Check</label>
          <span>{timings.grounding_ms?.toFixed(1) || 0} ms</span>
        </div>
        <div className="metric-pill" style={{ borderColor: 'var(--accent-pink)', background: 'rgba(236, 72, 153, 0.1)' }}>
          <label style={{ color: 'var(--accent-pink)' }}>Total Latency</label>
          <span style={{ color: 'var(--accent-pink)' }}>{timings.total_ms?.toFixed(1) || 0} ms</span>
        </div>
      </div>
    </div>
  );
}
