import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { dashboardApi } from '../../services/api';

const Progress = () => {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await dashboardApi.getStats();
        setStats(res);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <span className="loading loading-spinner loading-lg text-red-500"></span>
      </div>
    );
  }

  const scores = stats?.fluency_trend?.map((s: any) => s.score) || [0, 0, 0, 0, 0, 0, 0];

  return (
    <div className="max-w-5xl mx-auto space-y-12 pb-20">
      <div className="text-center mb-12">
        <h2 className="text-5xl font-black mb-4 tracking-tight text-text-primary">Fluency <span className="text-accent">Progress</span></h2>
        <p className="text-lg font-medium text-text-secondary">Track your clinical improvement with daily fluency analytics.</p>
      </div>

      <div className="p-10 rounded-[3rem] border border-border-subtle bg-card-bg shadow-premium transition-all duration-500">
        <div className="flex items-center justify-between mb-12">
          <h3 className="text-2xl font-black tracking-tight text-text-primary">Fluency Score Trend</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-accent shadow-[0_0_10px_rgba(56,178,172,0.5)]"></div>
              <span className="text-[10px] font-black uppercase tracking-widest text-text-secondary">Daily Average</span>
            </div>
          </div>
        </div>
        
        {/* SVG Graph */}
        <div className="h-[350px] w-full relative pt-10 px-4">
          <svg className="w-full h-full overflow-visible" viewBox={`0 0 ${(scores.length - 1) * 100} 300`} preserveAspectRatio="none">
            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map((level, i) => (
              <g key={i}>
                <line 
                  x1="0" y1={300 - (level * 3)} 
                  x2={(scores.length - 1) * 100} y2={300 - (level * 3)} 
                  stroke="var(--color-border-subtle)" 
                  strokeWidth="1" 
                  strokeDasharray="4 4"
                />
                <text 
                  x="-35" y={300 - (level * 3) + 4} 
                  fill="var(--color-text-secondary)" 
                  className="text-[10px] font-bold opacity-50"
                >
                  {level}%
                </text>
              </g>
            ))}

            {/* Area Fill */}
            <motion.path
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1.5, delay: 0.5 }}
              d={`
                M 0 300
                ${scores.map((s: number, i: number) => `L ${i * 100} ${300 - (s * 3)}`).join(' ')}
                L ${(scores.length - 1) * 100} 300
                Z
              `}
              fill="url(#gradient-accent)"
              className="opacity-10"
            />

            <defs>
              <linearGradient id="gradient-accent" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-accent)" />
                <stop offset="100%" stopColor="var(--color-accent)" stopOpacity="0" />
              </linearGradient>
            </defs>

            {/* Path */}
            <motion.path
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 2, ease: "easeInOut" }}
              d={scores.map((s: number, i: number) => `${i === 0 ? 'M' : 'L'} ${i * 100} ${300 - (s * 3)}`).join(' ')}
              fill="none"
              stroke="var(--color-accent)"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ filter: 'drop-shadow(0 10px 15px rgba(56,178,172,0.3))' }}
            />

            {/* Data points */}
            {scores.map((s: number, i: number) => (
              <g key={i}>
                <motion.circle
                  initial={{ r: 0 }}
                  animate={{ r: 6 }}
                  transition={{ delay: 1.5 + i * 0.1 }}
                  cx={i * 100}
                  cy={300 - (s * 3)}
                  fill="var(--color-accent)"
                  stroke="var(--color-card-bg)"
                  strokeWidth="3"
                  className="cursor-pointer hover:r-8 transition-all"
                />
                {/* Score label on hover or static */}
                <motion.text
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 2 + i * 0.1 }}
                  x={i * 100}
                  y={300 - (s * 3) - 15}
                  textAnchor="middle"
                  fill="var(--color-text-primary)"
                  className="text-[10px] font-black"
                >
                  {s}%
                </motion.text>
              </g>
            ))}
          </svg>
        </div>
        
        <div className="flex justify-between mt-12 px-4 border-t border-border-subtle pt-8">
          {stats?.fluency_trend?.map((s: any, i: number) => (
            <div key={i} className="flex flex-col items-center gap-1">
              <span className="text-[10px] font-black text-text-secondary uppercase tracking-widest">
                {s.day}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="p-8 rounded-[2.5rem] border border-border-subtle bg-card-bg shadow-premium transition-all duration-500 group">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 rounded-xl bg-accent/10 text-accent">
              <span className="text-xl">🏆</span>
            </div>
            <h4 className="text-sm font-black uppercase tracking-widest text-text-secondary">Milestones</h4>
          </div>
          <ul className="space-y-4">
            <li className="flex items-center gap-4">
              <div className="w-1.5 h-1.5 rounded-full bg-accent"></div>
              <span className="font-bold text-text-primary">Reached 80% Fluency</span>
              <span className="ml-auto text-[10px] text-text-secondary font-black uppercase">2 days ago</span>
            </li>
            <li className="flex items-center gap-4">
              <div className="w-1.5 h-1.5 rounded-full bg-accent"></div>
              <span className="font-bold text-text-primary">Completed 5-day Streak</span>
              <span className="ml-auto text-[10px] text-text-secondary font-black uppercase">Today</span>
            </li>
          </ul>
        </div>

        <div className="p-8 rounded-[2.5rem] border border-border-subtle bg-card-bg shadow-premium transition-all duration-500 group">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2.5 rounded-xl bg-orange-500/10 text-orange-500">
              <span className="text-xl">🎯</span>
            </div>
            <h4 className="text-sm font-black uppercase tracking-widest text-text-secondary">Focus Area</h4>
          </div>
          <p className="text-base mb-6 leading-relaxed text-text-secondary">
            Based on your last 3 sessions, you should focus on <strong className="text-text-primary">Light Contacts</strong> for plosive sounds (P, B, T).
          </p>
          <button className="text-accent font-black uppercase tracking-widest text-[10px] hover:text-accent-hover transition-colors flex items-center gap-2 group/btn">
            View suggested exercises 
            <span className="group-hover/btn:translate-x-1 transition-transform">→</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Progress;
