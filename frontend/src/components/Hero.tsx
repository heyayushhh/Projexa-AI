import { motion } from 'framer-motion';
import { useThemeStore } from '../store/themeStore';

interface HeroProps {
  onStart: () => void;
}

const Hero = ({ onStart }: HeroProps) => {
  const { isDark } = useThemeStore();

  return (
    <div className="w-full flex flex-col justify-center min-h-[90vh] px-4 sm:px-6 md:px-8 lg:px-10">
      <motion.div
        initial={{ opacity: 0, x: -30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="w-full max-w-2xl"
      >
        <h1 className={`text-4xl sm:text-5xl md:text-7xl lg:text-8xl font-bold leading-[1.08] sm:leading-[1.1] mb-7 sm:mb-10 md:mb-12 tracking-tight transition-colors duration-500 break-words ${
          isDark ? 'text-white' : 'text-[#111]'
        }`}>
          <span className="block">Analyze Your</span>
          <span className="block">Speech with</span>
          <span className="block bg-clip-text text-transparent bg-gradient-to-r from-[#FF2E2E] to-[#FF8080]">AI Precision</span>
        </h1>
        
        <p className={`text-sm sm:text-base md:text-lg lg:text-2xl mb-8 sm:mb-10 md:mb-14 max-w-[600px] leading-relaxed transition-colors duration-500 font-medium break-words ${
          isDark ? 'text-gray-300' : 'text-[#444]'
        }`}>
          Medical-grade deep learning technology designed to track, analyze, and help improve speech patterns with clinical accuracy.
        </p>
        
        <motion.button
          whileHover={{ scale: 1.05, boxShadow: isDark ? '0 20px 40px -10px rgba(255,46,46,0.4)' : '0 20px 40px -10px rgba(255,46,46,0.2)' }}
          whileTap={{ scale: 0.96 }}
          onClick={onStart}
          className="w-full sm:w-auto min-h-[44px] bg-[#FF2E2E] hover:bg-[#E02929] text-white font-bold py-4 sm:py-5 px-8 sm:px-12 md:px-14 rounded-full text-base sm:text-lg md:text-xl shadow-lg sm:shadow-2xl shadow-red-500/30 transition-all duration-300 border-none cursor-pointer"
        >
          Let's Start
        </motion.button>
      </motion.div>
    </div>
  );
};

export default Hero;
