// API Service for EMORA Frontend
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

async function fetchWithFallback(endpoint, options = {}) {
  const timeoutMs = options.timeout || 12000;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  const fetchOptions = {
    ...options,
    signal: controller.signal,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  };

  try {
    // Attempt direct API_BASE call
    const res = await fetch(`${API_BASE}${endpoint}`, fetchOptions);
    clearTimeout(id);
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error (${res.status})`);
    }
    return await res.json();
  } catch (err) {
    clearTimeout(id);
    // If direct fails (e.g. CORS or network), try proxy fallback if relative path exists
    if (!endpoint.startsWith('http') && API_BASE.includes('127.0.0.1')) {
      try {
        const fallbackRes = await fetch(`/api${endpoint}`, {
          ...options,
          headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        });
        if (fallbackRes.ok) {
          return await fallbackRes.json();
        }
      } catch {
        // Fall back to original error
      }
    }
    throw err;
  }
}

export async function getHealth() {
  try {
    return await fetchWithFallback('/health', { method: 'GET', timeout: 6000 });
  } catch (err) {
    console.warn('Health check failed:', err.message);
    return {
      status: 'offline',
      error: err.message,
    };
  }
}

export async function getSamples() {
  return await fetchWithFallback('/samples', { method: 'GET', timeout: 15000 });
}

export async function predictSegment(segmentId) {
  return await fetchWithFallback('/predict', {
    method: 'POST',
    body: JSON.stringify({ segment_id: segmentId }),
    timeout: 30000,
  });
}

export async function predictRaw(text, audioFile, videoFile) {
  const formData = new FormData();
  if (text) formData.append('text', text);
  if (audioFile) formData.append('audio', audioFile);
  if (videoFile) formData.append('video', videoFile);

  const timeoutMs = 120000; // 2 min for model inference
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}/predict/raw`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      // Do NOT set Content-Type header - browser sets it with boundary
    });
    clearTimeout(id);
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error (${res.status})`);
    }
    return await res.json();
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}
