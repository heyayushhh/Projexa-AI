import axios from 'axios';
import { useThemeStore } from '../store/themeStore';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add a request interceptor to include the JWT token
api.interceptors.request.use((config) => {
  const token = useThemeStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
    console.log("[DEBUG] Token attached to request:", config.url);
  } else {
    console.warn("[DEBUG] No token found for request:", config.url);
  }
  return config;
});

// Add a response interceptor to handle 401 errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      console.error("[DEBUG] 401 Unauthorized detected. Token may be invalid or expired.");
      // Optional: useThemeStore.getState().logout(); 
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  register: async (userData: any) => {
    const response = await api.post('/auth/register', userData);
    return response.data;
  },
  login: async (credentials: any) => {
    const response = await api.post('/auth/login', credentials);
    return response.data;
  },
};

export const dashboardApi = {
  getStats: async () => {
    const response = await api.get('/dashboard/stats');
    return response.data;
  },
  getSessions: async () => {
    const response = await api.get('/dashboard/sessions');
    return response.data;
  },
  analyze: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/dashboard/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },
  completeTraining: async (exerciseId: number, sessionData?: any) => {
    const response = await api.post(`/dashboard/training/complete?exercise_id=${exerciseId}`, sessionData);
    return response.data;
  },
  getTrainingProgress: async () => {
    const response = await api.get('/dashboard/training/progress');
    return response.data;
  },
};

export default api;
