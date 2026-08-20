const API_BASE = 'https://voice-rag-goa-visora.onrender.com';

export async function fetchHealth() {
  const resp = await fetch(`${API_BASE}/health`);

  if (!resp.ok) {
    throw new Error('Backend health check failed');
  }

  return await resp.json();
}

export async function fetchConfig() {
  const resp = await fetch(`${API_BASE}/api/config`);

  if (!resp.ok) {
    throw new Error('Failed to fetch backend configuration');
  }

  return await resp.json();
}

export async function sendTextQuery(queryText) {
  const resp = await fetch(`${API_BASE}/api/rag`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      query: queryText
    })
  });

  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));

    throw new Error(
      errorData.detail || 'RAG query failed'
    );
  }

  return await resp.json();
}

/**
 * Send recorded audio to the Voice RAG backend.
 *
 * IMPORTANT:
 * Do not rename the file to recording.wav.
 * The browser may have recorded WebM/Opus.
 * We preserve the real filename and MIME type.
 */
export async function sendVoiceQuery(audioFile) {
  if (!audioFile) {
    throw new Error('No audio file provided.');
  }

  if (!audioFile.size) {
    throw new Error('Audio file is empty.');
  }

  const formData = new FormData();

  formData.append(
    'file',
    audioFile,
    audioFile.name || 'recording.webm'
  );

  console.log('Sending voice file:', {
    name: audioFile.name,
    type: audioFile.type,
    size: audioFile.size
  });

  const resp = await fetch(`${API_BASE}/api/voice-rag`, {
    method: 'POST',
    body: formData
  });

  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));

    throw new Error(
      errorData.detail ||
      `Voice RAG request failed (${resp.status})`
    );
  }

  return await resp.json();
}

/**
 * Send audio only for testing Speech-To-Text.
 */
export async function sendSTTOnly(audioFile) {
  if (!audioFile) {
    throw new Error('No audio file provided.');
  }

  if (!audioFile.size) {
    throw new Error('Audio file is empty.');
  }

  const formData = new FormData();

  formData.append(
    'file',
    audioFile,
    audioFile.name || 'recording.webm'
  );

  console.log('Sending STT test file:', {
    name: audioFile.name,
    type: audioFile.type,
    size: audioFile.size
  });

  const resp = await fetch(`${API_BASE}/api/stt`, {
    method: 'POST',
    body: formData
  });

  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));

    throw new Error(
      errorData.detail ||
      `Speech-To-Text request failed (${resp.status})`
    );
  }

  return await resp.json();
}
