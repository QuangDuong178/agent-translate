import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

// ---- Health / System ----
export const getHealth = () => api.get('/health');
export const getSystemInfo = () => api.get('/system');

// ---- Models ----
export const listModels = () => api.get('/models');
export const downloadModel = (data) => api.post('/models/download', data);
export const deleteModel = (id) => api.delete(`/models/${id}`);

// ---- Datasets ----
export const listDatasets = () => api.get('/datasets');
export const downloadDataset = (data) => api.post('/datasets/download', data);
export const uploadDataset = (formData) =>
  api.post('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
export const deleteDataset = (id) => api.delete(`/datasets/${id}`);

// ---- Training ----
export const listTrainingJobs = () => api.get('/training');
export const getTrainingJob = (id) => api.get(`/training/${id}`);
export const startTraining = (data) => api.post('/training/start', data);
export const stopTraining = (id) => api.post(`/training/${id}/stop`);

// ---- Translation ----
export const translate = (data) => api.post('/translate', data);

// ---- Catalog ----
export const getCatalogModels = () => api.get('/catalog/models');
export const getCatalogDatasets = () => api.get('/catalog/datasets');

// ---- Languages ----
export const getLanguages = () => api.get('/languages');

// ---- Subtitles ----
export const listSubtitleJobs = () => api.get('/subtitles');
export const getSubtitleJob = (id) => api.get(`/subtitles/${id}`);
export const extractSubtitles = (data) => api.post('/subtitles/extract', data);
export const uploadVideoForSubtitles = (formData) =>
  api.post('/subtitles/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600000,
  });
export const translateSubtitles = (jobId, data) =>
  api.post(`/subtitles/${jobId}/translate`, data);
export const downloadSubtitleFile = (jobId, fileType) =>
  `${API_BASE}/api/subtitles/${jobId}/download/${fileType}`;

// ---- Settings ----
export const getSettings = () => api.get('/settings');
export const updateSettings = (data) => api.post('/settings', data);

// ---- Real-time Learning (Corrections) ----
export const submitCorrection = (data) => api.post('/corrections', data);
export const getCorrectionStats = () => api.get('/corrections/stats');
export const triggerCorrectionTraining = () => api.post('/corrections/train');

export default api;
