const API_BASE = 'https://voice-rag-goa-visora.onrender.com';

export async function fetchHealth() {
  const resp = await fetch(`${API_BASE}/health`);
  if (!resp.ok) throw new Error('Backend health check failed');
  return await resp.json();
}

export async function fetchConfig() {
  const resp = await fetch(`${API_BASE}/api/config`);
  if (!resp.ok) throw new Error('Failed to fetch backend configuration');
  return await resp.json();
}

export async function sendTextQuery(queryText) {
  const resp = await fetch(`${API_BASE}/api/rag`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: queryText })
  });
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || 'RAG query failed');
  }
  return await resp.json();
}

export async function sendVoiceQuery(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.wav');

  const resp = await fetch(`${API_BASE}/api/voice-rag`, {
    method: 'POST',
    body: formData
  });

  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Voice RAG request failed');
  }

  return await resp.json();
}

export async function sendSTTOnly(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.wav');

  const resp = await fetch(`${API_BASE}/api/stt`, {
    method: 'POST',
    body: formData
  });

  if (!resp.ok) throw new Error('Speech-To-Text request failed');
  return await resp.json();
}
