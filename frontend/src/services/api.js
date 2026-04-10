// src/services/api.js
// Central API client. All fetch calls go through here.

const BASE = process.env.REACT_APP_API_URL || '';

async function request(path, options = {}) {
  const token = localStorage.getItem('token');
  const headers = { 'Content-Type': 'application/json', ...options.headers };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw Object.assign(new Error(data.error || 'Request failed'), {
      status: res.status, data,
    });
  }
  return data;
}

export const api = {
  // Auth
  register: (email, username, password) =>
    request('/api/auth/register', { method: 'POST', body: JSON.stringify({ email, username, password }) }),

  login: (email, password) =>
    request('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),

  me: () => request('/api/auth/me'),

  // Analysis
  analyze: (text, language) =>
    request('/api/analyze', { method: 'POST', body: JSON.stringify({ text, language }) }),

  // Feedback
  submitFeedback: (payload) =>
    request('/api/submit-feedback', { method: 'POST', body: JSON.stringify(payload) }),

  // Admin
  history: (limit = 50) => request(`/api/history?limit=${limit}`),
  feedbackStats: () => request('/api/feedback/stats'),

  // Blockchain
  registerArticle: (text, language) =>
    request('/api/register-article', { method: 'POST', body: JSON.stringify({ text, language }) }),

  addAnnotation: (text, flagType, ipfsHash, confidence) =>
    request('/api/add-annotation', {
      method: 'POST',
      body: JSON.stringify({ text, flagType, ipfsHash, confidence }),
    }),

  health: () => request('/api/health'),
};
