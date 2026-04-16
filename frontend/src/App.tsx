import { useThemeStore } from './store/themeStore';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import AuthPage from './pages/AuthPage';
import Dashboard from './pages/Dashboard';
import DashboardMain from './pages/dashboard/DashboardMain';
import SpeechAnalysis from './pages/dashboard/SpeechAnalysis';
import Training from './pages/dashboard/Training';
import Progress from './pages/dashboard/Progress';
import { useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';

function App() {
  const { isDark, isAuthenticated } = useThemeStore();
  const location = useLocation();
  const navigate = useNavigate();

  // Fix: Unified theme and class application to the root element
  useEffect(() => {
    const root = document.documentElement;
    const theme = isDark ? 'dark' : 'light';
    root.setAttribute('data-theme', theme);
    if (isDark) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [isDark]);

  const handleStart = () => {
    if (isAuthenticated) {
      navigate('/dashboard');
    } else {
      navigate('/auth');
    }
  };

  const isDashboard = location.pathname.startsWith('/dashboard');

  return (
    <div className={`min-h-screen relative overflow-x-hidden transition-colors duration-500 ${
      isDark ? 'text-white' : 'text-[#111]'
    } ${isDashboard ? 'bg-bg-primary' : ''}`}>
      {/* Layer 1 (bottom): Background image → only for landing/auth */}
      {!isDashboard && (
        <div className="fixed inset-0 -z-20 pointer-events-none">
          <div
            className="absolute inset-0 bg-cover bg-center bg-no-repeat transition-all duration-700"
            style={{
              backgroundImage: `url(${isDark ? '/dark-bg.png' : '/light-bg.png'})`
            }}
          />
        </div>
      )}

      {/* Layer 2: Overlay → only for landing/auth */}
      {!isDashboard && (
        <div className="fixed inset-0 -z-10 pointer-events-none">
          {isDark ? (
            <div className="absolute inset-0 bg-black/60" />
          ) : (
            <div 
              className="absolute inset-0" 
              style={{ 
                background: 'linear-gradient(to right, rgba(255,255,255,0.7), rgba(255,255,255,0.3), transparent)',
                backgroundColor: 'rgba(255,255,255,0.15)'
              }} 
            />
          )}
        </div>
      )}

      {/* Layer 3: Main content → z-[10] */}
      <div className="relative z-10 flex flex-col min-h-screen">
        {!isDashboard && (
          <Navbar 
            onNavigate={(page) => {
              if (page === 'landing') navigate('/');
              if (page === 'auth') navigate('/auth');
              if (page === 'dashboard') navigate('/dashboard');
            }} 
            currentPage={isDashboard ? 'dashboard' : (location.pathname === '/' ? 'landing' : 'auth')}
          />
        )}
        
        <main className="flex-grow">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4 }}
              className="h-full"
            >
              <Routes location={location}>
                <Route path="/" element={<Landing onStart={handleStart} />} />
                <Route path="/auth" element={<AuthPage onAuthSuccess={() => {
                  navigate('/dashboard');
                }} />} />
                <Route path="/dashboard" element={
                  isAuthenticated ? <Dashboard /> : <Navigate to="/auth" />
                }>
                  <Route index element={<DashboardMain />} />
                  <Route path="analysis" element={<SpeechAnalysis />} />
                  <Route path="training" element={<Training />} />
                  <Route path="progress" element={<Progress />} />
                </Route>
                <Route path="*" element={<Navigate to="/" />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
