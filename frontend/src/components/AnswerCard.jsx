import React from 'react';
import { Bot, ShieldAlert, CheckCircle2 } from 'lucide-react';
import ConfidenceBadge from './ConfidenceBadge';

export default function AnswerCard({ answer, grounded, refused, confidence }) {
  if (!answer) return null;

  return (
    <div className="glass-panel" style={{ marginTop: '1rem' }}>
      <div className="card-header">
        <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Bot size={20} color="var(--accent-purple)" /> RAG Answer Response
        </h4>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <ConfidenceBadge score={confidence} />
          {refused ? (
            <span className="badge-tag" style={{ background: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24' }}>
              <ShieldAlert size={14} /> Guardrail Refusal
            </span>
          ) : grounded ? (
            <span className="badge-tag" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399' }}>
              <CheckCircle2 size={14} /> Grounded Answer
            </span>
          ) : (
            <span className="badge-tag" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#f87171' }}>
              Ungrounded / Unverified
            </span>
          )}
        </div>
      </div>
      <div className={`answer-box ${refused ? 'refusal' : ''}`}>
        <p>{answer}</p>
      </div>
    </div>
  );
}
