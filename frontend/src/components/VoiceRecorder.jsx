import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';

export default function VoiceRecorder({ onAudioRecorded, isLoading }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);

  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const timerRef = useRef(null);

  // Cleanup everything when component is removed
  useEffect(() => {
    return () => {
      cleanupRecording();
    };
  }, []);

  const cleanupRecording = () => {
    // Stop timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Stop waveform animation
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Stop microphone
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
      streamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      if (audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }

      audioContextRef.current = null;
    }

    analyserRef.current = null;
  };

  const getSupportedMimeType = () => {
    const mimeTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/ogg'
    ];

    for (const type of mimeTypes) {
      if (
        typeof MediaRecorder !== 'undefined' &&
        MediaRecorder.isTypeSupported(type)
      ) {
        return type;
      }
    }

    return '';
  };

  const getFileExtension = (mimeType) => {
    if (mimeType.includes('ogg')) {
      return 'ogg';
    }

    if (mimeType.includes('webm')) {
      return 'webm';
    }

    return 'webm';
  };

  const startRecording = async () => {
    try {
      // Browser support check
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert(
          'Your browser does not support microphone recording. Please use Chrome or Edge.'
        );
        return;
      }

      if (typeof MediaRecorder === 'undefined') {
        alert(
          'Audio recording is not supported by this browser. Please use Chrome or Edge.'
        );
        return;
      }

      // Stop anything left over from a previous recording
      cleanupRecording();

      // Request microphone with better speech-quality settings
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1
        }
      });

      streamRef.current = stream;

      // Log actual microphone settings
      const audioTrack = stream.getAudioTracks()[0];

      if (audioTrack) {
        console.log('Microphone:', audioTrack.label);
        console.log('Microphone settings:', audioTrack.getSettings());
      }

      // ----------------------------------------
      // Audio analyser for waveform visualization
      // ----------------------------------------

      const AudioContext =
        window.AudioContext || window.webkitAudioContext;

      if (AudioContext) {
        audioContextRef.current = new AudioContext();

        // Some browsers start AudioContext suspended
        if (audioContextRef.current.state === 'suspended') {
          await audioContextRef.current.resume();
        }

        const source =
          audioContextRef.current.createMediaStreamSource(stream);

        const analyser =
          audioContextRef.current.createAnalyser();

        analyser.fftSize = 64;
        analyser.smoothingTimeConstant = 0.8;

        source.connect(analyser);

        analyserRef.current = analyser;
      }

      // ----------------------------------------
      // Determine supported recording format
      // ----------------------------------------

      const mimeType = getSupportedMimeType();

      console.log(
        'Selected recording MIME type:',
        mimeType || 'browser default'
      );

      // ----------------------------------------
      // Create MediaRecorder
      // ----------------------------------------

      const recorderOptions = {};

      if (mimeType) {
        recorderOptions.mimeType = mimeType;
      }

      // Good quality for speech while keeping file size reasonable
      recorderOptions.audioBitsPerSecond = 128000;

      const mediaRecorder = new MediaRecorder(
        stream,
        recorderOptions
      );

      mediaRecorderRef.current = mediaRecorder;

      audioChunksRef.current = [];

      // ----------------------------------------
      // Collect audio data
      // ----------------------------------------

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // ----------------------------------------
      // Recording stopped
      // ----------------------------------------

      mediaRecorder.onstop = () => {
        try {
          const actualMimeType =
            mediaRecorder.mimeType ||
            mimeType ||
            'audio/webm';

          console.log(
            'Final recording MIME type:',
            actualMimeType
          );

          console.log(
            'Audio chunks:',
            audioChunksRef.current.length
          );

          // IMPORTANT:
          // Do NOT label WebM data as WAV.
          const audioBlob = new Blob(
            audioChunksRef.current,
            {
              type: actualMimeType
            }
          );

          console.log(
            'Recorded audio size:',
            audioBlob.size,
            'bytes'
          );

          if (audioBlob.size === 0) {
            console.error('Recorded audio is empty.');
            alert(
              'No audio was recorded. Please try again.'
            );

            cleanupRecording();
            return;
          }

          const extension =
            getFileExtension(actualMimeType);

          const filename =
            `recording.${extension}`;

          console.log(
            'Sending audio file:',
            filename
          );

          // Send audio to parent component
          onAudioRecorded(
            audioBlob,
            filename
          );

          // Cleanup microphone
          cleanupRecording();

          audioChunksRef.current = [];
        } catch (error) {
          console.error(
            'Error creating audio recording:',
            error
          );

          cleanupRecording();

          alert(
            'Could not process the recording. Please try again.'
          );
        }
      };

      // ----------------------------------------
      // Handle recording errors
      // ----------------------------------------

      mediaRecorder.onerror = (event) => {
        console.error(
          'MediaRecorder error:',
          event
        );

        cleanupRecording();

        setIsRecording(false);

        alert(
          'An error occurred while recording. Please try again.'
        );
      };

      // ----------------------------------------
      // Start recording
      // ----------------------------------------

      // Request data every 250ms.
      // This makes recording more reliable.
      mediaRecorder.start(250);

      setIsRecording(true);
      setRecordingTime(0);

      // ----------------------------------------
      // Recording timer
      // ----------------------------------------

      timerRef.current = setInterval(() => {
        setRecordingTime((previousTime) => {
          return previousTime + 1;
        });
      }, 1000);

      // ----------------------------------------
      // Waveform animation
      // ----------------------------------------

      startWaveformAnimation();

    } catch (error) {
      console.error(
        'Microphone access error:',
        error
      );

      setIsRecording(false);

      cleanupRecording();

      if (error?.name === 'NotAllowedError') {
        alert(
          'Microphone permission was denied. Please allow microphone access in your browser and try again.'
        );
      } else if (error?.name === 'NotFoundError') {
        alert(
          'No microphone was found. Please connect a microphone and try again.'
        );
      } else if (error?.name === 'NotReadableError') {
        alert(
          'The microphone is being used by another application. Close other apps using the microphone and try again.'
        );
      } else {
        alert(
          'Unable to access the microphone. Please check your microphone permissions and try again.'
        );
      }
    }
  };

  const startWaveformAnimation = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;

    if (!canvas || !analyser) {
      return;
    }

    const ctx = canvas.getContext('2d');

    if (!ctx) {
      return;
    }

    const bufferLength =
      analyser.frequencyBinCount;

    const dataArray =
      new Uint8Array(bufferLength);

    const drawWaveform = () => {
      if (
        !isRecording &&
        mediaRecorderRef.current?.state !== 'recording'
      ) {
        return;
      }

      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      const barWidth =
        (canvas.width / bufferLength) * 2.5;

      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const value = dataArray[i];

        const barHeight =
          (value / 255) * canvas.height;

        ctx.fillStyle =
          `rgb(${value + 100}, 99, 241)`;

        ctx.fillRect(
          x,
          canvas.height - barHeight,
          barWidth,
          barHeight
        );

        x += barWidth + 2;
      }

      animationFrameRef.current =
        requestAnimationFrame(drawWaveform);
    };

    drawWaveform();
  };

  const stopRecording = () => {
    const recorder =
      mediaRecorderRef.current;

    if (
      recorder &&
      recorder.state === 'recording'
    ) {
      console.log('Stopping recording...');

      recorder.stop();

      setIsRecording(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(
          animationFrameRef.current
        );

        animationFrameRef.current = null;
      }
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);

    const remainingSeconds =
      seconds % 60;

    return `${minutes}:${remainingSeconds
      .toString()
      .padStart(2, '0')}`;
  };

  return (
    <div className="glass-panel recorder-section">

      <h3>Voice Query Input</h3>

      <div className="mic-button-wrapper">

        <button
          type="button"
          className={`mic-button ${
            isRecording ? 'recording' : ''
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
            <Square size={36} />
          ) : (
            <Mic size={40} />
          )}

        </button>

      </div>

      <div style={{ textAlign: 'center' }}>

        <p
          style={{
            fontWeight: 600,
            color: isRecording
              ? '#ef4444'
              : 'var(--text-muted)'
          }}
        >

          {isRecording
            ? `Recording... ${formatTime(
                recordingTime
              )}`
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

    </div>
  );
}
