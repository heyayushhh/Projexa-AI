import axios, { type AxiosError, type InternalAxiosRequestConfig } from 'axios';
import { useThemeStore } from '../store/themeStore';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
});

api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = useThemeStore.getState().token;
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => Promise.reject(error)
);

export interface RegisterRequest {
  name: string;
  username: string;
  email: string;
  password: string;
  description?: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface AuthUser {
  id: string;
  username: string;
  name: string;
  email: string;
}

export interface AuthResponse {
  access_token: string;
  user: AuthUser;
}

export const authApi = {
  register: async (userData: RegisterRequest): Promise<unknown> => {
    const response = await api.post<unknown>('/auth/register', userData);
    return response.data;
  },
  login: async (credentials: LoginRequest): Promise<AuthResponse> => {
    const response = await api.post<AuthResponse>('/auth/login', credentials);
    return response.data;
  },
};

export interface DashboardStatsResponse {
  avg_score: number;
  streak: number;
  total_sessions: number;
  fluency_trend?: { day: string; score: number }[];
}

export interface DashboardSession {
  _id: string;
  created_at: string;
  session_type?: 'training' | 'analysis';
  level?: number;
  fluency_score: number;
  transcript?: string;
}

export const dashboardApi = {
  getStats: async (): Promise<DashboardStatsResponse> => {
    const response = await api.get<DashboardStatsResponse>('/dashboard/stats');
    return response.data;
  },
  getSessions: async (): Promise<DashboardSession[]> => {
    const response = await api.get<DashboardSession[]>('/dashboard/sessions');
    return response.data;
  },
  analyze: async (file: File): Promise<unknown> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<unknown>('/dashboard/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },
  completeTraining: async (exerciseId: number, sessionData?: Record<string, unknown>): Promise<unknown> => {
    const response = await api.post<unknown>(`/dashboard/training/complete?exercise_id=${exerciseId}`, sessionData);
    return response.data;
  },
  getTrainingProgress: async (): Promise<{ progress: Record<number, { score: number; is_correct: boolean }> }> => {
    const response = await api.get<{ progress: Record<number, { score: number; is_correct: boolean }> }>('/dashboard/training/progress');
    return response.data;
  },
};

export default api;
