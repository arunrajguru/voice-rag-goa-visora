import React from 'react';
import { FileText, Tag, Cpu } from 'lucide-react';

export default function TranscriptCard({ transcript, queryCategory, chunkStrategy }) {
  if (!transcript) return null;

  return (
    <div className="glass-panel" style={{ marginTop: '1rem' }}>
      <div className="card-header">
        <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FileText size={18} color="var(--accent-cyan)" /> Speech-To-Text Transcription
        </h4>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {queryCategory && (
            <span className="badge-tag" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#c084fc' }}>
              <Tag size={12} /> Category: {queryCategory}
            </span>
          )}
          {chunkStrategy && (
            <span className="badge-tag" style={{ background: 'rgba(6, 182, 212, 0.2)', color: '#22d3ee' }}>
              <Cpu size={12} /> Strategy: {chunkStrategy}
            </span>
          )}
        </div>
      </div>
      <p style={{ fontSize: '1.1rem', fontWeight: 500, color: 'var(--text-main)', background: 'rgba(0,0,0,0.2)', padding: '0.85rem 1.1rem', borderRadius: '0.5rem', borderLeft: '3px solid var(--accent-cyan)' }}>
        "{transcript}"
      </p>
    </div>
  );
}
