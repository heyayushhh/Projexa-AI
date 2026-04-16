import ThemeToggle from './ThemeToggle';
import { useThemeStore } from '../store/themeStore';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';

interface NavbarProps {
  onNavigate: (page: string) => void;
  currentPage: string;
}

const Navbar = ({ onNavigate, currentPage }: NavbarProps) => {
  const { isDark, isAuthenticated, logout } = useThemeStore();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/');
    onNavigate('landing');
  };

  const handleLogoClick = () => {
    navigate('/');
    onNavigate('landing');
  };

  const handleDashboardClick = () => {
    navigate('/dashboard');
    onNavigate('dashboard');
  };

  const handleAuthClick = () => {
    navigate('/auth');
    onNavigate('auth');
  };

  return (
    <nav className={`sticky top-0 z-50 border-b transition-all duration-500 ${
      isDark 
        ? 'bg-black/30 border-white/10 text-white' 
        : 'bg-white/30 border-black/5 text-[#111]'
    }`}>
      <div className="backdrop-blur-md">
        <div className="w-full px-4 sm:px-6 md:px-8 lg:px-10 py-3 sm:py-4 flex items-center justify-between gap-3">
          <div 
            className="text-lg sm:text-2xl font-bold cursor-pointer transition-colors duration-500 whitespace-nowrap"
            onClick={() => {
              setIsMenuOpen(false);
              handleLogoClick();
            }}
          >
            VocaCare
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-8">
              <div className={`flex items-center gap-6 transition-colors duration-500 ${isDark ? 'text-gray-300' : 'text-[#444]'}`}>
                <button className="hover:text-red-500 dark:hover:text-red-400 transition-colors cursor-pointer font-medium" onClick={() => onNavigate('why')}>Why</button>
                <button className="hover:text-red-500 dark:hover:text-red-400 transition-colors cursor-pointer font-medium" onClick={() => onNavigate('about')}>About</button>
                <button className="hover:text-red-500 dark:hover:text-red-400 transition-colors cursor-pointer font-medium" onClick={() => onNavigate('feedback')}>Feedback</button>
              </div>
              
              <div className="flex items-center gap-4 border-l border-black/10 dark:border-white/10 pl-8">
                {isAuthenticated ? (
                  <>
                    {currentPage !== 'dashboard' && (
                      <button 
                        className={`btn btn-sm rounded-full px-6 transition-all duration-300 shadow-md hover:scale-105 active:scale-95 ${
                          isDark ? 'btn-primary' : 'bg-[#FF2E2E] hover:bg-[#E02929] border-none text-white shadow-red-500/20'
                        }`}
                        onClick={handleDashboardClick}
                      >
                        Dashboard
                      </button>
                    )}
                    <button 
                      className={`btn btn-ghost btn-sm transition-colors duration-500 font-medium ${isDark ? 'text-gray-300' : 'text-[#444]'}`}
                      onClick={handleLogout}
                    >
                      Logout
                    </button>
                  </>
                ) : (
                  <>
                    {currentPage !== 'auth' && (
                      <button 
                        className={`btn btn-sm rounded-full px-6 transition-all duration-300 shadow-md hover:scale-105 active:scale-95 ${
                          isDark ? 'btn-primary' : 'bg-[#FF2E2E] hover:bg-[#E02929] border-none text-white shadow-red-500/20'
                        }`}
                        onClick={handleAuthClick}
                      >
                        LOGIN / SIGN IN
                      </button>
                    )}
                  </>
                )}
                
                <ThemeToggle />
              </div>
            </div>

            <div className="md:hidden flex items-center gap-2">
              <ThemeToggle />
              <button
                type="button"
                aria-label="Open menu"
                className={`min-h-[44px] min-w-[44px] rounded-xl border transition-colors ${
                  isDark ? 'border-white/10 bg-black/20 text-white' : 'border-black/10 bg-white/40 text-[#111]'
                }`}
                onClick={() => setIsMenuOpen((v) => !v)}
              >
                <span className="text-xl leading-none">☰</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {isMenuOpen && (
        <div className="md:hidden w-full px-4 sm:px-6 pb-4">
          <div className={`rounded-2xl border p-4 shadow-soft ${
            isDark ? 'bg-black/40 border-white/10' : 'bg-white/70 border-black/10'
          }`}>
            <div className={`flex flex-col gap-3 text-sm font-semibold ${isDark ? 'text-gray-200' : 'text-[#222]'}`}>
              <button className="text-left py-2" onClick={() => { setIsMenuOpen(false); onNavigate('why'); }}>Why</button>
              <button className="text-left py-2" onClick={() => { setIsMenuOpen(false); onNavigate('about'); }}>About</button>
              <button className="text-left py-2" onClick={() => { setIsMenuOpen(false); onNavigate('feedback'); }}>Feedback</button>
            </div>

            <div className="mt-4 flex flex-col gap-3">
              {isAuthenticated ? (
                <>
                  {currentPage !== 'dashboard' && (
                    <button
                      className="w-full min-h-[44px] bg-[#FF2E2E] hover:bg-[#E02929] text-white font-bold py-3 rounded-xl shadow-lg transition-all duration-300 border-none"
                      onClick={() => { setIsMenuOpen(false); handleDashboardClick(); }}
                    >
                      Dashboard
                    </button>
                  )}
                  <button
                    className={`w-full min-h-[44px] rounded-xl border font-bold py-3 transition-colors ${
                      isDark ? 'border-white/10 text-white bg-black/20' : 'border-black/10 text-[#111] bg-white/60'
                    }`}
                    onClick={() => { setIsMenuOpen(false); handleLogout(); }}
                  >
                    Logout
                  </button>
                </>
              ) : (
                <>
                  {currentPage !== 'auth' && (
                    <button
                      className="w-full min-h-[44px] bg-[#FF2E2E] hover:bg-[#E02929] text-white font-bold py-3 rounded-xl shadow-lg transition-all duration-300 border-none"
                      onClick={() => { setIsMenuOpen(false); handleAuthClick(); }}
                    >
                      Login / Sign in
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
