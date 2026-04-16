import React, { useEffect, useState } from 'react';
import { useThemeStore } from '../store/themeStore';
import Sidebar from '../components/Sidebar';
import { Outlet, useNavigate } from 'react-router-dom';
import BreathPacer from '../components/BreathPacer';

const Dashboard: React.FC = () => {
  const { isDark, user, token } = useThemeStore();
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    if (!token) {
      navigate('/auth');
    }
  }, [token, navigate]);

  if (!user) return null;

  return (
    <div className={`min-h-screen flex transition-colors duration-300 ${isDark ? 'dark bg-bg-primary text-text-primary' : 'bg-bg-primary text-text-primary'}`}>
      <div className={`md:hidden fixed inset-0 z-50 ${isSidebarOpen ? '' : 'pointer-events-none'}`}>
        <div
          className={`absolute inset-0 transition-opacity ${isSidebarOpen ? 'opacity-100' : 'opacity-0'} bg-black/50`}
          onClick={() => setIsSidebarOpen(false)}
        />
        <div className={`absolute inset-y-0 left-0 transition-transform duration-300 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
          <Sidebar
            className="md:hidden"
            showCloseButton
            onClose={() => setIsSidebarOpen(false)}
          />
        </div>
      </div>

      <Sidebar className="hidden md:flex sticky top-0" />

      {/* 2. Main Content Column */}
      <main className="flex-1 overflow-y-auto px-4 sm:px-6 md:px-8 py-6 sm:py-8 lg:p-12 relative z-10">
        <div className="max-w-5xl mx-auto">
          <div className="md:hidden sticky top-0 z-20 -mx-4 sm:-mx-6 px-4 sm:px-6 py-3 bg-bg-primary/90 backdrop-blur-xl border-b border-border-subtle">
            <div className="flex items-center justify-between gap-3">
              <button
                type="button"
                aria-label="Open navigation"
                className="min-h-[44px] min-w-[44px] rounded-xl border border-border-subtle bg-card-bg shadow-soft"
                onClick={() => setIsSidebarOpen(true)}
              >
                <span className="text-xl leading-none">☰</span>
              </button>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-bold uppercase tracking-widest text-text-secondary truncate">Welcome back</p>
                <p className="text-sm font-black text-text-primary truncate">{user.name || user.username}</p>
              </div>
            </div>
          </div>

          <header className="mb-6 sm:mb-10 flex flex-col sm:flex-row justify-between gap-4 sm:gap-6 sm:items-end">
            <div className="min-w-0">
              <p className="text-text-secondary font-medium mb-1 text-sm sm:text-base">Welcome back,</p>
              <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight break-words">{user.name || user.username}</h1>
            </div>
            <div className="text-left sm:text-right hidden md:block">
              <p className="text-xs font-bold uppercase tracking-widest text-text-secondary mb-1">Current Focus</p>
              <p className="text-accent font-bold">Daily Fluency Practice</p>
            </div>
          </header>
          
          <Outlet />
        </div>
      </main>

      {/* 3. Right Panel Column */}
      <aside className="w-80 hidden xl:block border-l border-border-subtle bg-bg-secondary p-8 space-y-8">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-widest text-text-secondary mb-6">Mindfulness</h3>
          <BreathPacer />
        </div>
        
        <div className="p-6 rounded-2xl bg-accent/5 border border-accent/10">
          <h4 className="font-bold mb-2">Pro Tip</h4>
          <p className="text-sm text-text-secondary leading-relaxed">
            Taking three deep breaths before an exercise can significantly reduce speech tension.
          </p>
        </div>
      </aside>
    </div>
  );
};

export default Dashboard;
