import React, { useState, useEffect, useRef } from 'react';

import Header from './components/Header';
import VoiceRecorder from './components/VoiceRecorder';
import PipelineStatus from './components/PipelineStatus';
import TranscriptCard from './components/TranscriptCard';
import AnswerCard from './components/AnswerCard';
import SourcesCard from './components/SourcesCard';
import LatencyCard from './components/LatencyCard';
import ExampleQuestions from './components/ExampleQuestions';

import {
  fetchHealth,
  sendVoiceQuery,
  sendTextQuery
} from './services/api';


export default function App() {

  // ==========================================================
  // STATE
  // ==========================================================

  const [isHealthy, setIsHealthy] = useState(false);

  const [isLoading, setIsLoading] = useState(false);

  const [currentStep, setCurrentStep] = useState('idle');

  const [pipelineResult, setPipelineResult] = useState(null);

  const [errorMsg, setErrorMsg] = useState(null);

  // Store progress timer IDs so they can be cancelled
  const progressTimersRef = useRef([]);


  // ==========================================================
  // CLEAR PROGRESS TIMERS
  // ==========================================================

  const clearProgressTimers = () => {

    progressTimersRef.current.forEach((timer) => {
      clearTimeout(timer);
    });

    progressTimersRef.current = [];
  };


  // ==========================================================
  // START PIPELINE VISUAL PROGRESS
  // ==========================================================

  const startProgressAnimation = (firstStep = 'voice') => {

    clearProgressTimers();

    setCurrentStep(firstStep);

    const steps = [
      {
        step: 'stt',
        delay: 300
      },
      {
        step: 'retrieve',
        delay: 900
      },
      {
        step: 'rerank',
        delay: 1500
      },
      {
        step: 'generate',
        delay: 2100
      },
      {
        step: 'verify',
        delay: 2700
      }
    ];

    steps.forEach(({ step, delay }) => {

      const timer = setTimeout(() => {
        setCurrentStep(step);
      }, delay);

      progressTimersRef.current.push(timer);
    });
  };


  // ==========================================================
  // CHECK BACKEND HEALTH
  // ==========================================================

  useEffect(() => {

    let mounted = true;

    const checkBackend = async () => {

      try {

        console.log(
          '[APP] Checking backend health...'
        );

        const result = await fetchHealth();

        console.log(
          '[APP] Backend health:',
          result
        );

        if (mounted) {
          setIsHealthy(
            result?.status === 'healthy'
          );
        }

      } catch (error) {

        console.error(
          '[APP] Backend health check failed:',
          error
        );

        if (mounted) {
          setIsHealthy(false);
        }
      }
    };

    checkBackend();

    return () => {
      mounted = false;
    };

  }, []);


  // ==========================================================
  // VOICE RECORDING COMPLETE
  // ==========================================================

  const handleAudioRecorded = async (audioFile) => {

    console.log(
      '================================================'
    );

    console.log(
      '[APP] AUDIO RECORDING RECEIVED'
    );

    console.log(
      '[APP] File:',
      audioFile
    );

    console.log(
      '[APP] Name:',
      audioFile?.name
    );

    console.log(
      '[APP] Type:',
      audioFile?.type
    );

    console.log(
      '[APP] Size:',
      audioFile?.size
    );

    console.log(
      '================================================'
    );


    // ----------------------------------------------------------
    // Validate audio
    // ----------------------------------------------------------

    if (!audioFile) {

      setErrorMsg(
        'No audio recording was received.'
      );

      setCurrentStep('idle');

      return;
    }


    if (audioFile.size === 0) {

      setErrorMsg(
        'The recorded audio file is empty. Please record again.'
      );

      setCurrentStep('idle');

      return;
    }


    // ----------------------------------------------------------
    // Start loading
    // ----------------------------------------------------------

    setIsLoading(true);

    setErrorMsg(null);

    setPipelineResult(null);


    // ----------------------------------------------------------
    // Show pipeline progress
    // ----------------------------------------------------------

    startProgressAnimation('voice');


    try {

      console.log(
        '[APP] Sending voice file to backend...'
      );

      console.log(
        '[APP] Endpoint:',
        'https://voice-rag-goa-visora.onrender.com/api/voice-rag'
      );


      // --------------------------------------------------------
      // SEND AUDIO TO BACKEND
      // --------------------------------------------------------

      const result = await sendVoiceQuery(
        audioFile
      );


      console.log(
        '[APP] VOICE RAG RESPONSE:',
        result
      );


      // --------------------------------------------------------
      // Validate response
      // --------------------------------------------------------

      if (!result) {

        throw new Error(
          'Backend returned an empty response.'
        );
      }


      // --------------------------------------------------------
      // Save result
      // --------------------------------------------------------

      setPipelineResult(result);

      clearProgressTimers();

      setCurrentStep('answer');


    } catch (error) {

      console.error(
        '================================================'
      );

      console.error(
        '[APP] VOICE RAG FAILED'
      );

      console.error(
        '[APP] Error:',
        error
      );

      console.error(
        '[APP] Message:',
        error?.message
      );

      console.error(
        '================================================'
      );


      clearProgressTimers();


      let message =
        error?.message ||
        'Voice RAG pipeline execution failed.';


      // --------------------------------------------------------
      // Better error messages
      // --------------------------------------------------------

      if (
        message.toLowerCase().includes('failed to fetch')
      ) {

        message =
          'Could not connect to the Voice RAG backend. Please check the Render backend and try again.';
      }


      if (
        message.toLowerCase().includes('cors')
      ) {

        message =
          'Backend CORS configuration is blocking the frontend request.';
      }


      setErrorMsg(message);

      setCurrentStep('idle');


    } finally {

      setIsLoading(false);
    }
  };


  // ==========================================================
  // EXAMPLE TEXT QUESTION
  // ==========================================================

  const handleSelectQuestion = async (queryText) => {

    console.log(
      '[APP] Example question selected:',
      queryText
    );


    if (!queryText || !queryText.trim()) {

      setErrorMsg(
        'The selected question is empty.'
      );

      return;
    }


    // ----------------------------------------------------------
    // Start loading
    // ----------------------------------------------------------

    setIsLoading(true);

    setErrorMsg(null);

    setPipelineResult(null);


    // ----------------------------------------------------------
    // Start progress animation
    // ----------------------------------------------------------

    startProgressAnimation('stt');


    try {

      console.log(
        '[APP] Sending text query to backend...'
      );


      const result = await sendTextQuery(
        queryText
      );


      console.log(
        '[APP] TEXT RAG RESPONSE:',
        result
      );


      if (!result) {

        throw new Error(
          'Backend returned an empty response.'
        );
      }


      setPipelineResult(result);

      clearProgressTimers();

      setCurrentStep('answer');


    } catch (error) {

      console.error(
        '[APP] TEXT RAG ERROR:',
        error
      );


      clearProgressTimers();


      setErrorMsg(
        error?.message ||
        'Text RAG query execution failed.'
      );


      setCurrentStep('idle');


    } finally {

      setIsLoading(false);
    }
  };


  // ==========================================================
  // COMPONENT CLEANUP
  // ==========================================================

  useEffect(() => {

    return () => {

      clearProgressTimers();

    };

  }, []);


  // ==========================================================
  // UI
  // ==========================================================

  return (

    <div className="app-container">


      {/* ====================================================
          HEADER
      ==================================================== */}

      <Header
        isHealthy={isHealthy}
      />


      {/* ====================================================
          PIPELINE STATUS
      ==================================================== */}

      <PipelineStatus
        currentStep={currentStep}
        isCompleted={!!pipelineResult}
      />


      {/* ====================================================
          MAIN GRID
      ==================================================== */}

      <div className="main-grid">


        {/* ==================================================
            LEFT SIDE
        ================================================== */}

        <div>

          {/* ------------------------------------------------
              VOICE RECORDER
          ------------------------------------------------ */}

          <VoiceRecorder
            onAudioRecorded={
              handleAudioRecorded
            }
            isLoading={isLoading}
          />


          {/* ------------------------------------------------
              EXAMPLE QUESTIONS
          ------------------------------------------------ */}

          <ExampleQuestions
            onSelectQuestion={
              handleSelectQuestion
            }
          />

        </div>


        {/* ==================================================
            RIGHT SIDE
        ================================================== */}

        <div>


          {/* =================================================
              ERROR
          ================================================= */}

          {errorMsg && (

            <div
              className="glass-panel"
              style={{
                borderColor: 'var(--danger)',
                color: '#f87171',
                marginBottom: '1rem'
              }}
            >

              <strong>
                Error:
              </strong>

              {' '}

              {errorMsg}

            </div>

          )}


          {/* =================================================
              RESULT
          ================================================= */}

          {pipelineResult ? (

            <>

              {/* ---------------------------------------------
                  TRANSCRIPT
              --------------------------------------------- */}

              <TranscriptCard
                transcript={
                  pipelineResult.transcript
                }

                queryCategory={
                  pipelineResult.query_category
                }

                chunkStrategy={
                  pipelineResult.chunk_strategy
                }
              />


              {/* ---------------------------------------------
                  ANSWER
              --------------------------------------------- */}

              <AnswerCard
                answer={
                  pipelineResult.answer
                }

                grounded={
                  pipelineResult.grounded
                }

                refused={
                  pipelineResult.refused
                }

                confidence={
                  pipelineResult.confidence
                }
              />


              {/* ---------------------------------------------
                  LATENCY
              --------------------------------------------- */}

              <LatencyCard
                timings={
                  pipelineResult.timings
                }
              />


              {/* ---------------------------------------------
                  SOURCES
              --------------------------------------------- */}

              <SourcesCard
                sources={
                  pipelineResult.sources
                }
              />

            </>

          ) : (

            /* =================================================
               EMPTY STATE
            ================================================= */

            <div
              className="glass-panel"
              style={{
                textAlign: 'center',
                padding: '3rem 1.5rem',
                color: 'var(--text-muted)'
              }}
            >

              <p>

                Record a voice query or click
                an example prompt to view
                real-time RAG execution &
                latency breakdown.

              </p>

            </div>

          )}

        </div>

      </div>

    </div>

  );
}
