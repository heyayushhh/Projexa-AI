import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { dashboardApi } from '../../services/api';
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
import { Link } from 'react-router-dom';

const DashboardMain: React.FC = () => {
  const { isDark } = useThemeStore();
  const [stats, setStats] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, sessionsRes] = await Promise.all([
          dashboardApi.getStats(),
          dashboardApi.getSessions(),
        ]);
        
        // Map backend response fields to frontend state
        setStats({
          fluencyScore: statsRes.avg_score,
          currentStreak: statsRes.streak,
          totalSessions: statsRes.total_sessions
        });
        setSessions(sessionsRes);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="w-10 h-10 border-4 border-accent border-t-transparent rounded-full animate-spin"></div>
    </div>
  );

  return (
    <div className="space-y-10">
      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard 
          icon={TrendingUp} 
          label="Fluency Score" 
          value={`${stats?.fluencyScore || 0}%`} 
          color="text-accent"
          isDark={isDark}
        />
        <StatCard 
          icon={Flame} 
          label="Current Streak" 
          value={`${stats?.currentStreak || 0} Days`} 
          color="text-orange-400"
          isDark={isDark}
        />
        <StatCard 
          icon={Activity} 
          label="Total Sessions" 
          value={stats?.totalSessions || 0} 
          color="text-blue-400"
          isDark={isDark}
        />
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
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
                className="p-5 rounded-2xl border border-border-subtle bg-card-bg shadow-soft hover:shadow-premium group transition-all duration-300 flex items-center justify-between"
              >
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors duration-300 ${
                    session.session_type === 'training' 
                      ? 'bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white' 
                      : 'bg-blue-500/10 text-blue-500 group-hover:bg-blue-500 group-hover:text-white'
                  }`}>
                    {session.session_type === 'training' ? <Trophy size={20} /> : <Mic2 size={20} />}
                  </div>
                  <div>
                    <p className="font-bold text-text-primary">
                      {session.session_type === 'training' ? `Level ${session.level}` : 'Speech Analysis'}
                    </p>
                    <p className="text-xs text-text-secondary flex items-center gap-1">
                      <Clock size={12} /> {new Date(session.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
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

const StatCard = ({ icon: Icon, label, value, color }: any) => (
  <div className="p-6 rounded-[2rem] border border-border-subtle bg-card-bg shadow-soft hover:shadow-premium transition-all duration-500 group">
    <div className="flex items-center justify-between mb-4">
      <div className={`p-3 rounded-2xl bg-bg-primary group-hover:scale-110 transition-transform duration-500 ${color}`}>
        <Icon size={24} />
      </div>
    </div>
    <p className="text-sm font-bold text-text-secondary uppercase tracking-widest mb-1">{label}</p>
    <p className="text-3xl font-black tracking-tight text-text-primary">{value}</p>
  </div>
);

const ActionButton = ({ to, icon: Icon, label, sub }: any) => (
  <Link to={to} className="flex items-center gap-4 p-5 rounded-2xl border border-border-subtle bg-card-bg shadow-soft hover:shadow-premium hover:border-accent/30 transition-all duration-300 group">
    <div className="p-3 rounded-xl bg-accent/10 text-accent group-hover:bg-accent group-hover:text-white transition-all duration-300">
      <Icon size={20} />
    </div>
    <div>
      <p className="font-bold text-sm text-text-primary group-hover:text-accent transition-colors">{label}</p>
      <p className="text-xs text-text-secondary">{sub}</p>
    </div>
  </Link>
);

export default DashboardMain;
