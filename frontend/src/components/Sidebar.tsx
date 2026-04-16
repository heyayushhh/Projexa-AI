import React from 'react';
import { motion } from 'framer-motion';
import { 
  LayoutDashboard, 
  Mic2, 
  TrendingUp, 
  Trophy,
  LogOut,
  Sun,
  Moon
} from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { useThemeStore } from '../store/themeStore';

interface SidebarProps {
  className?: string;
  onNavigate?: () => void;
  onClose?: () => void;
  showCloseButton?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ className, onNavigate, onClose, showCloseButton }) => {
  const { isDark, toggleTheme, user, logout } = useThemeStore();
  const location = useLocation();

  const menuItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard, path: '/dashboard' },
    { id: 'analysis', label: 'Analysis', icon: Mic2, path: '/dashboard/analysis' },
    { id: 'training', label: 'Training', icon: Trophy, path: '/dashboard/training' },
    { id: 'progress', label: 'Insights', icon: TrendingUp, path: '/dashboard/progress' },
  ];

  const isActive = (path: string) => {
    if (path === '/dashboard') return location.pathname === '/dashboard';
    return location.pathname.startsWith(path);
  };

  return (
    <aside className={`w-72 flex flex-col border-r border-border-subtle bg-sidebar-bg h-screen transition-colors duration-300 ${className || ''}`}>
      <div className="p-6 sm:p-8">
        <div className="flex items-center justify-between gap-3 mb-10 sm:mb-12 px-4">
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-accent flex items-center justify-center shadow-lg shadow-accent/20 shrink-0">
              <span className="text-white font-bold text-lg sm:text-xl">V</span>
            </div>
            <span className="text-lg sm:text-2xl font-bold tracking-tight text-text-primary whitespace-nowrap leading-none">
              VocaCare
            </span>
          </div>
          {showCloseButton && (
            <button
              type="button"
              aria-label="Close menu"
              className={`min-h-[44px] min-w-[44px] rounded-xl border transition-colors ${
                isDark ? 'border-white/10 bg-black/20 text-white' : 'border-black/10 bg-white/60 text-[#111]'
              }`}
              onClick={onClose}
            >
              <span className="text-xl leading-none">✕</span>
            </button>
          )}
        </div>

        <nav className="space-y-1">
          {menuItems.map((item) => {
            const active = isActive(item.path);
            return (
              <Link
                key={item.id}
                to={item.path}
                onClick={() => {
                  onNavigate?.();
                  onClose?.();
                }}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group relative ${
                  active 
                    ? 'bg-sidebar-active text-accent' 
                    : 'text-text-secondary hover:bg-bg-primary/50 hover:text-text-primary'
                }`}
              >
                <div className={`p-2 rounded-lg transition-colors duration-300 ${
                  active ? 'bg-accent/10 text-accent' : 'group-hover:bg-accent/5'
                }`}>
                  <item.icon size={20} strokeWidth={active ? 2.5 : 2} />
                </div>
                <span className={`font-semibold text-sm tracking-tight transition-colors duration-300 ${
                  active ? 'text-accent' : 'text-text-secondary group-hover:text-text-primary'
                }`}>
                  {item.label}
                </span>
                
                {active && (
                  <motion.div 
                    layoutId="sidebarActive"
                    className="absolute left-0 w-1.5 h-6 bg-accent rounded-r-full shadow-[2px_0_10px_rgba(56,178,172,0.4)]"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  />
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="mt-auto p-6 sm:p-8 space-y-2">
        <div className="px-4 py-4 rounded-2xl bg-bg-primary/30 border border-border-subtle mb-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-full bg-accent/20 flex items-center justify-center text-accent font-bold">
              {user?.name?.[0] || user?.username?.[0] || 'U'}
            </div>
            <div className="overflow-hidden">
              <p className="text-sm font-bold text-text-primary truncate">{user?.name || user?.username}</p>
              <p className="text-[10px] font-bold text-text-secondary uppercase tracking-widest">Premium Member</p>
            </div>
          </div>
        </div>

        <button
          onClick={toggleTheme}
          className="w-full min-h-[44px] flex items-center gap-3 px-4 py-3 rounded-xl text-text-secondary hover:bg-bg-primary hover:text-text-primary transition-all duration-300 group"
        >
          <div className="p-2 rounded-lg group-hover:bg-accent/5 transition-colors">
            {isDark ? <Sun size={18} /> : <Moon size={18} />}
          </div>
          <span className="font-semibold text-sm">{isDark ? 'Light Mode' : 'Dark Mode'}</span>
        </button>

        <button
          onClick={() => {
            logout();
            onClose?.();
          }}
          className="w-full min-h-[44px] flex items-center gap-3 px-4 py-3 rounded-xl text-text-secondary hover:bg-red-500/5 hover:text-red-500 transition-all duration-300 group"
        >
          <div className="p-2 rounded-lg group-hover:bg-red-500/10 transition-colors">
            <LogOut size={18} />
          </div>
          <span className="font-semibold text-sm">Sign Out</span>
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
