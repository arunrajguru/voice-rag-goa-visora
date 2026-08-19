import React from 'react';
import { Mic, Zap, ShieldCheck } from 'lucide-react';

export default function Header({ isHealthy }) {
  return (
    <header className="header-hero">
      <div className="badge-tag">
        <Zap size={14} /> HH Goa 2026 Task 2: Voice-Enabled Adaptive RAG
      </div>
      <h1 className="gradient-title">Voice-Enabled Adaptive RAG</h1>
      <p className="header-subtitle">
        Low-latency retrieval-augmented generation engine powered by Sarvam AI STT, FAISS in-memory vector search, BM25, & multi-guardrail verification.
      </p>
      <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.25rem' }}>
        <span className="badge-tag" style={{ background: isHealthy ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)', color: isHealthy ? '#34d399' : '#f87171' }}>
          <ShieldCheck size={14} /> {isHealthy ? 'Backend Active' : 'Backend Degraded'}
        </span>
        <span className="badge-tag" style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee' }}>
          Dataset: ai4bharat/MSMARCO-XI
        </span>
      </div>
    </header>
  );
}
