import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';

export default function VoiceRecorder({ onAudioRecorded, isLoading }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [error, setError] = useState('');

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);

  const animationFrameRef = useRef(null);
  const timerRef = useRef(null);

  const canvasRef = useRef(null);

  // --------------------------------------------------
  // Find the best audio format supported by browser
  // --------------------------------------------------
  const getSupportedMimeType = () => {
    const mimeTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
      'audio/wav'
    ];

    for (const mimeType of mimeTypes) {
      if (
        typeof MediaRecorder !== 'undefined' &&
        MediaRecorder.isTypeSupported(mimeType)
      ) {
        return mimeType;
      }
    }

    return '';
  };

  // --------------------------------------------------
  // Start waveform animation
  // --------------------------------------------------
  const startVisualizer = (analyser) => {
    const canvas = canvasRef.current;

    if (!canvas || !analyser) return;

    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    analyser.fftSize = 256;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!analyserRef.current) return;

      analyser.getByteTimeDomainData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2;

      ctx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);

      ctx.strokeStyle = '#8b5cf6';
      ctx.stroke();

      animationFrameRef.current =
        requestAnimationFrame(draw);
    };

    draw();
  };

  // --------------------------------------------------
  // Stop visualizer
  // --------------------------------------------------
  const stopVisualizer = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    const canvas = canvasRef.current;

    if (canvas) {
      const ctx = canvas.getContext('2d');

      if (ctx) {
        ctx.clearRect(
          0,
          0,
          canvas.width,
          canvas.height
        );
      }
    }
  };

  // --------------------------------------------------
  // Start recording
  // --------------------------------------------------
  const startRecording = async () => {
    setError('');

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          'Microphone recording is not supported by this browser.'
        );
      }

      // Request microphone with speech-friendly processing
      const stream =
        await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 16000
          }
        });

      streamRef.current = stream;

      // ----------------------------------------------
      // Audio context for visualizer
      // ----------------------------------------------
      const AudioContext =
        window.AudioContext ||
        window.webkitAudioContext;

      if (AudioContext) {
        const audioContext = new AudioContext();

        audioContextRef.current = audioContext;

        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }

        const source =
          audioContext.createMediaStreamSource(stream);

        const analyser =
          audioContext.createAnalyser();

        analyser.smoothingTimeConstant = 0.75;

        source.connect(analyser);

        sourceRef.current = source;
        analyserRef.current = analyser;

        startVisualizer(analyser);
      }

      // ----------------------------------------------
      // Select supported recording format
      // ----------------------------------------------
      const mimeType = getSupportedMimeType();

      if (!mimeType) {
        throw new Error(
          'No supported audio recording format found.'
        );
      }

      console.log(
        'Recording MIME type:',
        mimeType
      );

      // ----------------------------------------------
      // Create MediaRecorder
      // ----------------------------------------------
      const recorderOptions = {
        mimeType
      };

      // Opus quality
      if (mimeType.includes('webm')) {
        recorderOptions.audioBitsPerSecond = 128000;
      }

      const mediaRecorder =
        new MediaRecorder(
          stream,
          recorderOptions
        );

      mediaRecorderRef.current = mediaRecorder;

      audioChunksRef.current = [];

      // ----------------------------------------------
      // Receive audio chunks
      // ----------------------------------------------
      mediaRecorder.ondataavailable = (event) => {
        if (
          event.data &&
          event.data.size > 0
        ) {
          audioChunksRef.current.push(
            event.data
          );
        }
      };

      // ----------------------------------------------
      // Recording finished
      // ----------------------------------------------
      mediaRecorder.onstop = async () => {
        try {
          // Give browser time to flush final chunk
          const actualMimeType =
            mediaRecorder.mimeType ||
            mimeType;

          const audioBlob =
            new Blob(
              audioChunksRef.current,
              {
                type: actualMimeType
              }
            );

          console.log(
            'Recorded audio:',
            {
              size: audioBlob.size,
              type: audioBlob.type
            }
          );

          if (audioBlob.size === 0) {
            setError(
              'No audio was recorded. Please try again.'
            );
            return;
          }

          // ------------------------------------------
          // IMPORTANT:
          // Keep the REAL file format.
          // Do NOT call it recording.wav if WebM.
          // ------------------------------------------
          let filename = 'recording.webm';

          if (actualMimeType.includes('ogg')) {
            filename = 'recording.ogg';
          } else if (
            actualMimeType.includes('mp4')
          ) {
            filename = 'recording.m4a';
          } else if (
            actualMimeType.includes('wav')
          ) {
            filename = 'recording.wav';
          }

          // Create a File instead of anonymous Blob
          const audioFile =
            new File(
              [audioBlob],
              filename,
              {
                type: actualMimeType
              }
            );

          console.log(
            'Uploading audio:',
            {
              name: audioFile.name,
              type: audioFile.type,
              size: audioFile.size
            }
          );

          // Send to parent
          onAudioRecorded(audioFile);

        } catch (err) {
          console.error(
            'Audio processing error:',
            err
          );

          setError(
            'Failed to process the recording.'
          );
        } finally {
          cleanupAudio();
        }
      };

      mediaRecorder.onerror = (event) => {
        console.error(
          'MediaRecorder error:',
          event
        );

        setError(
          'Recording failed. Please try again.'
        );

        cleanupAudio();
        setIsRecording(false);
      };

      // ----------------------------------------------
      // Start recorder
      //
      // timeslice = 250ms
      // This makes Chrome continuously generate
      // chunks instead of waiting until the end.
      // ----------------------------------------------
      mediaRecorder.start(250);

      setIsRecording(true);
      setRecordingTime(0);

      // ----------------------------------------------
      // Recording timer
      // ----------------------------------------------
      timerRef.current = setInterval(() => {
        setRecordingTime(
          (previous) => previous + 1
        );
      }, 1000);

    } catch (err) {
      console.error(
        'Microphone error:',
        err
      );

      let message =
        'Unable to access microphone.';

      if (err?.name === 'NotAllowedError') {
        message =
          'Microphone permission was denied. Please allow microphone access in your browser.';
      } else if (
        err?.name === 'NotFoundError'
      ) {
        message =
          'No microphone was found on this device.';
      } else if (
        err?.name === 'NotReadableError'
      ) {
        message =
          'Microphone is already being used by another application.';
      } else if (err?.message) {
        message = err.message;
      }

      setError(message);

      cleanupAudio();
      setIsRecording(false);
    }
  };

  // --------------------------------------------------
  // Stop recording
  // --------------------------------------------------
  const stopRecording = () => {
    const recorder =
      mediaRecorderRef.current;

    if (
      !recorder ||
      recorder.state === 'inactive'
    ) {
      return;
    }

    setIsRecording(false);

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // IMPORTANT:
    // Request the final data chunk before stopping.
    if (recorder.state === 'recording') {
      recorder.requestData();
      recorder.stop();
    }
  };

  // --------------------------------------------------
  // Cleanup audio resources
  // --------------------------------------------------
  const cleanupAudio = () => {
    // Timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Visualizer
    stopVisualizer();

    // Stop microphone
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    // Disconnect audio source
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch (e) {
        // Ignore
      }

      sourceRef.current = null;
    }

    analyserRef.current = null;

    // Close AudioContext
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (e) {
        // Ignore
      }

      audioContextRef.current = null;
    }
  };

  // --------------------------------------------------
  // Cleanup when component unmounts
  // --------------------------------------------------
  useEffect(() => {
    return () => {
      const recorder =
        mediaRecorderRef.current;

      if (
        recorder &&
        recorder.state !== 'inactive'
      ) {
        recorder.stop();
      }

      cleanupAudio();
    };
  }, []);

  // --------------------------------------------------
  // UI
  // --------------------------------------------------
  return (
    <div className="glass-panel recorder-section">

      <h3>Voice Query Input</h3>

      <div className="mic-button-wrapper">

        <button
          type="button"
          className={`mic-button ${
            isRecording
              ? 'recording'
              : ''
          }`}
          onClick={
            isRecording
              ? stopRecording
              : startRecording
          }
          disabled={isLoading}
          aria-label={
            isRecording
              ? 'Stop recording'
              : 'Start recording'
          }
        >

          {isLoading ? (
            <Loader2
              className="animate-spin"
              size={40}
            />
          ) : isRecording ? (
            <Square
              size={36}
              fill="currentColor"
            />
          ) : (
            <Mic size={40} />
          )}

        </button>

      </div>

      <div
        style={{
          textAlign: 'center'
        }}
      >

        <p
          style={{
            fontWeight: 600,
            color: isRecording
              ? '#ef4444'
              : 'var(--text-muted)'
          }}
        >
          {isRecording
            ? `Recording... (${recordingTime}s)`
            : isLoading
              ? 'Processing Voice RAG Pipeline...'
              : 'Click Mic to Record Query'}
        </p>

      </div>

      <canvas
        ref={canvasRef}
        className="waveform-canvas"
        width={300}
        height={40}
      />

      {error && (
        <div
          style={{
            marginTop: '12px',
            padding: '10px 14px',
            borderRadius: '10px',
            background: 'rgba(239, 68, 68, 0.12)',
            color: '#ef4444',
            fontSize: '14px',
            textAlign: 'center'
          }}
        >
          {error}
        </div>
      )}

    </div>
  );
}
