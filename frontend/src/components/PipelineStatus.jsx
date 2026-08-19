import React from 'react';
import { Mic, FileText, Search, Filter, Cpu, CheckCircle, Award } from 'lucide-react';

const STEPS = [
  { id: 'voice', label: 'Voice', icon: Mic },
  { id: 'stt', label: 'Sarvam STT', icon: FileText },
  { id: 'retrieve', label: 'Hybrid Retrieve', icon: Search },
  { id: 'rerank', label: 'Rerank', icon: Filter },
  { id: 'generate', label: 'LLM Gen', icon: Cpu },
  { id: 'verify', label: 'Ground Verify', icon: CheckCircle },
  { id: 'answer', label: 'Answer', icon: Award }
];

export default function PipelineStatus({ currentStep, isCompleted }) {
  return (
    <div className="glass-panel" style={{ padding: '1rem 1.5rem' }}>
      <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        End-to-End Pipeline Execution Stepper
      </h4>
      <div className="pipeline-stepper">
        {STEPS.map((step, idx) => {
          const Icon = step.icon;
          const isActive = currentStep === step.id;
          const isDone = isCompleted || STEPS.findIndex(s => s.id === currentStep) > idx;

          return (
            <div key={step.id} className={`step-item ${isActive ? 'active' : ''} ${isDone ? 'completed' : ''}`}>
              <div className="step-circle">
                <Icon size={16} />
              </div>
              <span>{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
