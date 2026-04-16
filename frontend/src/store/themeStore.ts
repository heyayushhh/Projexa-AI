import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: string; // 🔥 Changed from str to string
  username: string;
  name: string;
  email: string;
}

interface ThemeState {
  isDark: boolean;
  toggleTheme: () => void;
  isAuthenticated: boolean;
  token: string | null;
  user: User | null;
  login: (token: string, user: User) => void;
  logout: () => void;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      isDark: false,
      toggleTheme: () => set((state) => ({ isDark: !state.isDark })),
      isAuthenticated: false,
      token: null,
      user: null,
      login: (token, user) => set({ 
        isAuthenticated: true, 
        token, 
        user 
      }),
      logout: () => set({ 
        isAuthenticated: false, 
        token: null, 
        user: null 
      }),
    }),
    {
      name: 'vocacare-storage',
    }
  )
);
