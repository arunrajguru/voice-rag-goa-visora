import React from 'react';

export default function ConfidenceBadge({ score }) {
  if (score === undefined || score === null) return null;
  const pct = Math.round(score * 100);
  
  let color = '#10b981';
  if (pct < 40) color = '#ef4444';
  else if (pct < 70) color = '#f59e0b';

  return (
    <span className="badge-tag" style={{ background: `${color}22`, color: color, borderColor: `${color}44` }}>
      Confidence: {pct}%
    </span>
  );
}
