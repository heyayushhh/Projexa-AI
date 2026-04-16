import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const BreathPacer: React.FC = () => {
  const [phase, setPhase] = useState<'Inhale' | 'Hold' | 'Exhale'>('Inhale');
  const [scale, setScale] = useState(1);

  useEffect(() => {
    let isMounted = true;
    
    const cycle = async () => {
      if (!isMounted) return;

      // Inhale: 4s
      setPhase('Inhale');
      setScale(1.8);
      await new Promise(r => setTimeout(r, 4000));
      if (!isMounted) return;

      // Hold: 2s
      setPhase('Hold');
      await new Promise(r => setTimeout(r, 2000));
      if (!isMounted) return;

      // Exhale: 4s
      setPhase('Exhale');
      setScale(1);
      await new Promise(r => setTimeout(r, 4000));
      if (!isMounted) return;

      cycle(); // Repeat
    };

    cycle();
    return () => { isMounted = false; };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center p-8 rounded-[2rem] bg-card-bg border border-border-subtle shadow-soft transition-colors duration-300">
      <div className="relative flex items-center justify-center w-40 h-40 mb-8">
        {/* Pulsing Outer Rings */}
        <motion.div
          animate={{ scale: scale }}
          transition={{ duration: 4, ease: "easeInOut" }}
          className="absolute w-20 h-20 rounded-full bg-accent/10 border border-accent/20"
        />
        <motion.div
          animate={{ scale: scale * 0.8 }}
          transition={{ duration: 4, ease: "easeInOut" }}
          className="absolute w-20 h-20 rounded-full bg-accent/20"
        />
        
        {/* Inner Core */}
        <motion.div
          animate={{ 
            scale: scale * 0.5,
            backgroundColor: phase === 'Hold' ? 'var(--accent)' : 'var(--accent)',
            opacity: phase === 'Hold' ? 0.8 : 1
          }}
          transition={{ duration: 4, ease: "easeInOut" }}
          className="absolute w-20 h-20 rounded-full bg-accent shadow-[0_0_30px_rgba(56,178,172,0.4)] flex items-center justify-center"
        >
          <span className="text-white font-black text-[10px] uppercase tracking-tighter drop-shadow-md">
            {phase}
          </span>
        </motion.div>
      </div>
      
      <div className="text-center space-y-1">
        <p className="text-[10px] font-black text-accent uppercase tracking-widest">Mindful Breathing</p>
        <p className="text-[11px] font-medium text-text-secondary leading-tight max-w-[140px] mx-auto">
          Sync your breath to calm your speech center.
        </p>
      </div>
    </div>
  );
};

export default BreathPacer;