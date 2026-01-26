/**
 * API client for Behavior Scope backend
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Fetch all sessions
 * @param {number} limit - Maximum number of sessions
 * @param {number} offset - Number of sessions to skip
 * @returns {Promise<Array>} List of sessions
 */
export const fetchSessions = async (limit = 100, offset = 0) => {
  const response = await api.get('/sessions', {
    params: { limit, offset }
  });
  return response.data;
};

/**
 * Fetch a specific session
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Session data
 */
export const fetchSession = async (sessionId) => {
  const response = await api.get(`/sessions/${sessionId}`);
  return response.data;
};

/**
 * Fetch frames for a session
 * @param {string} sessionId - Session identifier
 * @param {Object} options - Query options
 * @returns {Promise<Array>} Frame data
 */
export const fetchSessionFrames = async (sessionId, options = {}) => {
  const response = await api.get(`/sessions/${sessionId}/frames`, {
    params: options
  });
  return response.data;
};

/**
 * Fetch aggregate statistics
 * @returns {Promise<Object>} Statistics data
 */
export const fetchStatistics = async () => {
  const response = await api.get('/statistics');
  return response.data;
};

/**
 * Fetch detailed eye contact audit for a session
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Eye contact audit data
 */
export const fetchEyeContactAudit = async (sessionId) => {
  const response = await api.get(`/sessions/${sessionId}/eye-contact-audit`);
  return response.data;
};

/**
 * Search sessions by criteria
 * @param {Object} criteria - Search criteria
 * @returns {Promise<Array>} Matching sessions
 */
export const searchSessions = async (criteria) => {
  const response = await api.get('/search', {
    params: criteria
  });
  return response.data;
};

/**
 * Fetch stuttering analysis audit for a session
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Stuttering audit data with calculation breakdown
 */
export const fetchStutteringAudit = async (sessionId) => {
  const response = await api.get(`/sessions/${sessionId}/clinical/stuttering`);
  return response.data;
};

/**
 * Fetch turn-taking analysis audit for a session
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Turn-taking audit data with calculation breakdown
 */
export const fetchTurnTakingAudit = async (sessionId) => {
  const response = await api.get(`/sessions/${sessionId}/autism/turn-taking`);
  return response.data;
};

/**
 * Fetch question-response analysis audit for a session
 * @param {string} sessionId - Session identifier
 * @returns {Promise<Object>} Responsiveness audit data with calculation breakdown
 */
export const fetchResponsivenessAudit = async (sessionId) => {
  const response = await api.get(`/sessions/${sessionId}/clinical/responsiveness`);
  return response.data;
};

export default api;

