import { motion } from 'framer-motion';
import { useThemeStore } from '../store/themeStore';

const ThemeToggle = () => {
  const { isDark, toggleTheme } = useThemeStore();

  return (
    <motion.div 
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={`flex items-center cursor-pointer w-16 h-8 rounded-full p-1 transition-all duration-500 relative ${
        isDark 
          ? 'bg-indigo-900 shadow-[0_0_15px_rgba(99,102,241,0.4)]' 
          : 'bg-amber-100 shadow-[0_0_15px_rgba(251,191,36,0.3)]'
      }`}
      onClick={toggleTheme}
    >
      <div className="absolute inset-0 flex justify-between items-center px-2 pointer-events-none text-xs">
        <span className={isDark ? 'opacity-30' : 'opacity-100'}>☀️</span>
        <span className={isDark ? 'opacity-100' : 'opacity-30'}>🌙</span>
      </div>
      <motion.div
        className={`w-6 h-6 rounded-full shadow-lg z-10 ${
          isDark ? 'bg-indigo-200' : 'bg-amber-400'
        }`}
        animate={{ x: isDark ? 32 : 0 }}
        transition={{ 
          type: "spring", 
          stiffness: 400, 
          damping: 25,
          mass: 1
        }}
      />
    </motion.div>
  );
};

export default ThemeToggle;
