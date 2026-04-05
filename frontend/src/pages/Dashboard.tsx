import React, { useEffect } from 'react';
import { useThemeStore } from '../store/themeStore';
import Sidebar from '../components/Sidebar';
import { Outlet, useNavigate } from 'react-router-dom';
import BreathPacer from '../components/BreathPacer';

const Dashboard: React.FC = () => {
  const { isDark, user, token } = useThemeStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (!token) {
      navigate('/auth');
    }
  }, [token, navigate]);

  if (!user) return null;

  return (
    <div className={`min-h-screen flex transition-colors duration-300 ${isDark ? 'dark bg-bg-primary text-text-primary' : 'bg-bg-primary text-text-primary'}`}>
      {/* 1. Sidebar Column */}
      <Sidebar />

      {/* 2. Main Content Column */}
      <main className="flex-1 overflow-y-auto p-8 lg:p-12 relative z-10">
        <div className="max-w-5xl mx-auto">
          <header className="mb-10 flex justify-between items-end">
            <div>
              <p className="text-text-secondary font-medium mb-1">Welcome back,</p>
              <h1 className="text-4xl font-bold tracking-tight">{user.name || user.username}</h1>
            </div>
            <div className="text-right hidden md:block">
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
