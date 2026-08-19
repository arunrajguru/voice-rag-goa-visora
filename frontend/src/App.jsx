import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import VoiceRecorder from './components/VoiceRecorder';
import PipelineStatus from './components/PipelineStatus';
import TranscriptCard from './components/TranscriptCard';
import AnswerCard from './components/AnswerCard';
import SourcesCard from './components/SourcesCard';
import LatencyCard from './components/LatencyCard';
import ExampleQuestions from './components/ExampleQuestions';

import { fetchHealth, sendVoiceQuery, sendTextQuery } from './services/api';

export default function App() {
  const [isHealthy, setIsHealthy] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState('idle');
  const [pipelineResult, setPipelineResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    fetchHealth()
      .then(res => setIsHealthy(res.status === 'healthy'))
      .catch(() => setIsHealthy(false));
  }, []);

  const handleAudioRecorded = async (audioBlob) => {
    setIsLoading(true);
    setErrorMsg(null);
    setCurrentStep('voice');

    try {
      setTimeout(() => setCurrentStep('stt'), 200);
      setTimeout(() => setCurrentStep('retrieve'), 500);
      setTimeout(() => setCurrentStep('rerank'), 800);
      setTimeout(() => setCurrentStep('generate'), 1100);
      setTimeout(() => setCurrentStep('verify'), 1400);

      const result = await sendVoiceQuery(audioBlob);
      setPipelineResult(result);
      setCurrentStep('answer');
    } catch (err) {
      setErrorMsg(err.message || 'Voice RAG pipeline execution error');
      setCurrentStep('idle');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectQuestion = async (queryText) => {
    setIsLoading(true);
    setErrorMsg(null);
    setCurrentStep('stt');

    try {
      setTimeout(() => setCurrentStep('retrieve'), 200);
      setTimeout(() => setCurrentStep('rerank'), 400);
      setTimeout(() => setCurrentStep('generate'), 600);
      setTimeout(() => setCurrentStep('verify'), 800);

      const result = await sendTextQuery(queryText);
      setPipelineResult(result);
      setCurrentStep('answer');
    } catch (err) {
      setErrorMsg(err.message || 'Text RAG query execution error');
      setCurrentStep('idle');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header isHealthy={isHealthy} />

      <PipelineStatus currentStep={currentStep} isCompleted={!!pipelineResult} />

      <div className="main-grid">
        <div>
          <VoiceRecorder onAudioRecorded={handleAudioRecorded} isLoading={isLoading} />
          <ExampleQuestions onSelectQuestion={handleSelectQuestion} />
        </div>

        <div>
          {errorMsg && (
            <div className="glass-panel" style={{ borderColor: 'var(--danger)', color: '#f87171' }}>
              <strong>Error:</strong> {errorMsg}
            </div>
          )}

          {pipelineResult ? (
            <>
              <TranscriptCard
                transcript={pipelineResult.transcript}
                queryCategory={pipelineResult.query_category}
                chunkStrategy={pipelineResult.chunk_strategy}
              />
              <AnswerCard
                answer={pipelineResult.answer}
                grounded={pipelineResult.grounded}
                refused={pipelineResult.refused}
                confidence={pipelineResult.confidence}
              />
              <LatencyCard timings={pipelineResult.timings} />
              <SourcesCard sources={pipelineResult.sources} />
            </>
          ) : (
            <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem 1.5rem', color: 'var(--text-muted)' }}>
              <p>Record a voice query or click an example prompt to view real-time RAG execution & latency breakdown.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
