import React from 'react';
import { Database, FileCode } from 'lucide-react';

export default function SourcesCard({ sources }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="glass-panel" style={{ marginTop: '1rem' }}>
      <div className="card-header">
        <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Database size={18} color="var(--accent-pink)" /> Retrieved Context Sources ({sources.length})
        </h4>
      </div>
      <div>
        {sources.map((src, idx) => (
          <div key={idx} className="source-item">
            <div className="source-meta">
              <span><strong>Doc ID:</strong> {src.document_id}</span>
              <span>•</span>
              <span><strong>Chunk ID:</strong> {src.chunk_id}</span>
              <span>•</span>
              <span><strong>Strategy:</strong> {src.strategy}</span>
              <span>•</span>
              <span><strong>Score:</strong> {(src.score * 100).toFixed(1)}%</span>
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
              {src.text}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
