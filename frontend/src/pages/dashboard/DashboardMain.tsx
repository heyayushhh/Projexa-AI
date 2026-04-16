import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { dashboardApi, type DashboardSession } from '../../services/api';
import { useThemeStore } from '../../store/themeStore';
import { 
  Activity, 
  Flame, 
  Mic2, 
  TrendingUp, 
  Clock,
  ArrowRight,
  Trophy
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Link } from 'react-router-dom';

interface DashboardStatsUi {
  fluencyScore: number;
  currentStreak: number;
  totalSessions: number;
}

const DashboardMain: React.FC = () => {
  const { token } = useThemeStore();
  const [stats, setStats] = useState<DashboardStatsUi | null>(null);
  const [sessions, setSessions] = useState<DashboardSession[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    const fetchData = async () => {
      setLoading(true);
      try {
        const [statsRes, sessionsRes] = await Promise.all([
          dashboardApi.getStats(),
          dashboardApi.getSessions(),
        ]);
        
        // Map backend response fields to frontend state
        if (!cancelled) {
          setStats({
            fluencyScore: statsRes.avg_score,
            currentStreak: statsRes.streak,
            totalSessions: statsRes.total_sessions
          });
          setSessions(sessionsRes);
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchData();
    return () => {
      cancelled = true;
    };
  }, [token]);

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="w-10 h-10 border-4 border-accent border-t-transparent rounded-full animate-spin"></div>
    </div>
  );

  return (
    <div className="space-y-6 sm:space-y-10">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
        <StatCard 
          icon={TrendingUp} 
          label="Fluency Score" 
          value={`${stats?.fluencyScore || 0}%`} 
          color="text-accent"
        />
        <StatCard 
          icon={Flame} 
          label="Current Streak" 
          value={`${stats?.currentStreak || 0} Days`} 
          color="text-orange-400"
        />
        
        <StatCard 
          icon={Activity} 
          label="Total Sessions" 
          value={stats?.totalSessions || 0} 
          color="text-blue-400"
        />

      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
        {/* Activity Feed */}
        <div className="lg:col-span-2 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-bold tracking-tight text-text-primary">Recent Activity</h3>
            <Link to="/dashboard/progress" className="text-sm font-bold text-accent hover:text-accent-hover transition-colors flex items-center gap-1 group">
              View All <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
          
          <div className="space-y-4">
            {sessions.map((session, i) => (
              <motion.div
                key={session._id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="p-4 sm:p-5 rounded-2xl border border-border-subtle bg-card-bg shadow-soft hover:shadow-premium group transition-all duration-300 flex items-center justify-between gap-4"
              >
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors duration-300 ${
                    session.session_type === 'training' 
                      ? 'bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white' 
                      : 'bg-blue-500/10 text-blue-500 group-hover:bg-blue-500 group-hover:text-white'
                  }`}>
                    {session.session_type === 'training' ? <Trophy size={20} /> : <Mic2 size={20} />}
                  </div>
                  <div className="min-w-0">
                    <p className="font-bold text-text-primary">
                      {session.session_type === 'training' ? `Level ${session.level}` : 'Speech Analysis'}
                    </p>
                    <p className="text-xs text-text-secondary flex items-center gap-1">
                      <Clock size={12} /> {new Date(session.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-lg font-black text-text-primary">{session.fluency_score}%</p>
                  <p className="text-[10px] font-bold uppercase tracking-widest text-text-secondary">Fluency</p>
                </div>
              </motion.div>
            ))}
            {sessions.length === 0 && (
              <div className="text-center py-12 rounded-3xl border-2 border-dashed border-border-subtle bg-bg-primary/50">
                <p className="text-text-secondary font-medium">No activity yet. Start your first session!</p>
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="space-y-6">
          <h3 className="text-xl font-bold tracking-tight text-text-primary">Quick Actions</h3>
          <div className="space-y-3">
            <ActionButton 
              to="/dashboard/training" 
              icon={Trophy} 
              label="Continue Training" 
              sub="Master the roadmap" 
            />
            <ActionButton 
              to="/dashboard/analysis" 
              icon={Mic2} 
              label="New Analysis" 
              sub="Check current fluency" 
            />
          </div>
        </div>
      </div>
    </div>
  );
};

interface StatCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  color: string;
}

const StatCard = ({ icon: Icon, label, value, color }: StatCardProps) => (
  <div className="p-4 sm:p-6 rounded-[2rem] border border-border-subtle bg-card-bg shadow-soft hover:shadow-premium transition-all duration-500 group">
    <div className="flex items-center justify-between mb-4">
      <div className={`p-3 rounded-2xl bg-bg-primary group-hover:scale-110 transition-transform duration-500 ${color}`}>
        <Icon size={24} />
      </div>
    </div>
    <p className="text-sm font-bold text-text-secondary uppercase tracking-widest mb-1">{label}</p>
    <p className="text-2xl sm:text-3xl font-black tracking-tight text-text-primary">{value}</p>
  </div>
);

interface ActionButtonProps {
  to: string;
  icon: LucideIcon;
  label: string;
  sub: string;
}

const ActionButton = ({ to, icon: Icon, label, sub }: ActionButtonProps) => (
  <Link to={to} className="flex items-center gap-4 p-4 sm:p-5 rounded-2xl border border-border-subtle bg-card-bg shadow-soft hover:shadow-premium hover:border-accent/30 transition-all duration-300 group">
    <div className="p-3 rounded-xl bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white transition-all duration-300">
      <Icon size={20} />
    </div>
    <div className="min-w-0">
      <p className="font-bold text-sm text-text-primary group-hover:text-accent transition-colors">{label}</p>
      <p className="text-xs text-text-secondary break-words">{sub}</p>
    </div>
  </Link>
);

export default DashboardMain;
