import { motion, AnimatePresence } from 'framer-motion';
import { dashboardApi } from '../../services/api';
import { useState, useEffect } from 'react';
import { EXERCISES } from './exerciseData';
import type { Exercise } from './exerciseData';
import SpeechPractice from './SpeechPractice';

const Training = () => {
  const [progress, setProgress] = useState<Record<number, { score: number; is_correct: boolean }>>({});
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null);
  const [activeTab, setActiveTab] = useState<Exercise['type']>('SoundRep');

  useEffect(() => {
    const fetchProgress = async () => {
      try {
        const data = await dashboardApi.getTrainingProgress();
        setProgress(data.progress || {});
      } catch (err) {
        console.error("Failed to fetch progress:", err);
      }
    };
    fetchProgress();
  }, []);

  const handleComplete = async (exercise: Exercise, sessionData: any) => {
    try {
      console.log("[DEBUG] Completing exercise:", exercise.id);
      await dashboardApi.completeTraining(exercise.id, {
        ...sessionData,
        type: exercise.type,
        difficulty: exercise.difficulty,
        level: exercise.level
      });
      
      // Update local progress with the new score
      const exerciseId = Number(exercise.id);
      const updatedProgress = {
        ...progress,
        [exerciseId]: {
          score: sessionData.fluencyScore,
          is_correct: sessionData.isCorrect
        }
      };
      
      console.log("[DEBUG] Updated progress map:", updatedProgress);
      setProgress(updatedProgress);
      
      // Auto-open next level if current is completed successfully
      if (sessionData.isCorrect) {
        const nextLevel = categoryExercises.find(e => 
          (e.difficulty === exercise.difficulty && e.level === exercise.level + 1) ||
          (exercise.level === 10 && exercise.difficulty === 'Easy' && e.difficulty === 'Medium' && e.level === 1) ||
          (exercise.level === 10 && exercise.difficulty === 'Medium' && e.difficulty === 'Hard' && e.level === 1)
        );
        
        if (nextLevel) {
          console.log("[DEBUG] Unlocking next level:", nextLevel.id);
          setTimeout(() => {
            setSelectedExercise(nextLevel);
          }, 800);
        } else {
          setSelectedExercise(null);
        }
      } else {
        setSelectedExercise(null);
      }
    } catch (err: any) {
      console.error("[ERROR] Failed to save training session:", err);
      if (err.response?.status === 401) {
        alert("Your session has expired. Please log out and log in again to save your progress.");
      } else {
        alert("Failed to save progress. Please check your internet connection.");
      }
    }
  };

  const categoryExercises = EXERCISES.filter(ex => ex.type === activeTab);
  
  // Group by difficulty
  const groupedExercises = {
    Easy: categoryExercises.filter(ex => ex.difficulty === 'Easy').sort((a, b) => a.level - b.level),
    Medium: categoryExercises.filter(ex => ex.difficulty === 'Medium').sort((a, b) => a.level - b.level),
    Hard: categoryExercises.filter(ex => ex.difficulty === 'Hard').sort((a, b) => a.level - b.level)
  };

  const categories: { type: Exercise['type']; label: string; icon: string }[] = [
    { type: 'SoundRep', label: 'Sound Rep', icon: '🔤' },
    { type: 'WordRep', label: 'Word Rep', icon: '🗣️' },
    { type: 'Interjection', label: 'Interject', icon: '💭' },
    { type: 'Prolongation', label: 'Prolong', icon: '〰️' },
    { type: 'NoStutteredWords', label: 'Fluency', icon: '✨' },
  ];

  const isUnlocked = (ex: Exercise) => {
    // Level 1 of Easy is always unlocked
    if (ex.level === 1 && ex.difficulty === 'Easy') return true;
    
    // Rule: Previous level in the SAME category (activeTab) must be completed with is_correct: true
    const categoryExercises = EXERCISES.filter(e => e.type === activeTab);
    
    // Find the exercise that comes immediately before this one in the roadmap
    let prevEx: Exercise | undefined;

    if (ex.level > 1) {
      // Level 2-10: previous level in same difficulty
      prevEx = categoryExercises.find(e => e.difficulty === ex.difficulty && e.level === ex.level - 1);
    } else if (ex.difficulty === 'Medium') {
      // Medium Level 1: needs Easy Level 10
      prevEx = categoryExercises.find(e => e.difficulty === 'Easy' && e.level === 10);
    } else if (ex.difficulty === 'Hard') {
      // Hard Level 1: needs Medium Level 10
      prevEx = categoryExercises.find(e => e.difficulty === 'Medium' && e.level === 10);
    }
    
    if (!prevEx) return false;

    // Check if the previous exercise exists in the progress map and is marked as correct
    const prevId = Number(prevEx.id);
    const prevProgress = progress[prevId];
    return prevProgress ? prevProgress.is_correct === true : false;
  };

  return (
    <div className="max-w-6xl mx-auto space-y-12 pb-24">
      <div className="text-center mb-12">
        <h2 className="text-5xl font-black mb-4 tracking-tight text-text-primary">
          Training <span className="text-accent">Roadmap</span>
        </h2>
        <p className="text-lg font-medium text-text-secondary max-w-2xl mx-auto">
          Progress through clinical exercises designed to build speech confidence and muscle memory.
        </p>
      </div>

      {/* Category Tabs */}
      <div className="flex flex-wrap justify-center gap-3 mb-16 p-2 rounded-3xl bg-bg-primary/50 border border-border-subtle inline-flex mx-auto w-full max-w-fit">
        {categories.map((cat) => (
          <button
            key={cat.type}
            onClick={() => setActiveTab(cat.type)}
            className={`px-6 py-3 rounded-2xl font-bold uppercase tracking-widest text-[10px] transition-all duration-300 flex items-center gap-2 ${
              activeTab === cat.type
                ? 'bg-accent text-white shadow-lg shadow-accent/20'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-primary'
            }`}
          >
            <span className="text-base">{cat.icon}</span>
            {cat.label}
          </button>
        ))}
      </div>

      <div className="space-y-24">
        {(['Easy', 'Medium', 'Hard'] as const).map((difficulty) => (
          <div key={difficulty} className="relative">
            <div className="flex items-center gap-6 mb-12">
              <div className={`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-[0.2em] ${
                difficulty === 'Easy' ? 'bg-green-500/10 text-green-500' : 
                difficulty === 'Medium' ? 'bg-blue-500/10 text-blue-500' : 'bg-purple-500/10 text-purple-500'
              }`}>
                {difficulty} Phase
              </div>
              <div className="h-px flex-1 bg-gradient-to-r from-border-subtle to-transparent"></div>
            </div>

            <div className="flex flex-wrap justify-center gap-8 md:gap-14 relative">
              {groupedExercises[difficulty].map((ex) => {
                const unlocked = isUnlocked(ex);
                const levelProgress = progress[ex.id];
                const isDone = levelProgress?.is_correct;
                const pastScore = levelProgress?.score;
                
                return (
                  <motion.div
                    key={ex.id}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    whileHover={unlocked ? { y: -8, scale: 1.05 } : {}}
                    className="relative z-10"
                  >
                    <button
                      onClick={() => unlocked && setSelectedExercise(ex)}
                      disabled={!unlocked}
                      className={`w-20 h-20 rounded-[2rem] flex flex-col items-center justify-center transition-all duration-500 border-2 relative group ${
                        isDone 
                          ? 'bg-accent border-accent text-white shadow-premium' 
                          : unlocked 
                            ? 'bg-card-bg border-accent/30 text-accent hover:border-accent shadow-soft' 
                            : 'bg-bg-primary/50 border-border-subtle text-text-secondary/30 grayscale'
                      }`}
                    >
                      <span className="text-[10px] font-black mb-0.5 opacity-60">LVL</span>
                      <span className="text-2xl font-black tracking-tighter">{ex.level}</span>
                      
                      {isDone && (
                        <div className="absolute -top-2 -right-2 w-7 h-7 bg-white rounded-full flex items-center justify-center text-accent shadow-premium border-2 border-accent transform group-hover:scale-110 transition-transform">
                          <span className="font-black text-sm">✓</span>
                        </div>
                      )}
                      
                      {!unlocked && (
                        <div className="absolute -top-2 -right-2 w-7 h-7 bg-bg-secondary rounded-full flex items-center justify-center text-text-secondary/40 shadow-soft">
                          <span className="text-xs">🔒</span>
                        </div>
                      )}

                      {/* Score Badge */}
                      {pastScore !== undefined && (
                        <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 bg-white/90 dark:bg-black/90 backdrop-blur-md border border-border-subtle px-2.5 py-0.5 rounded-full z-20 shadow-soft">
                          <span className={`text-[9px] font-black ${pastScore >= 70 ? 'text-accent' : 'text-orange-500'}`}>
                            {pastScore}%
                          </span>
                        </div>
                      )}
                    </button>
                    
                    {unlocked && !isDone && (
                      <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 whitespace-nowrap">
                        <span className="text-[9px] font-black uppercase tracking-widest text-accent animate-pulse">
                          Start
                        </span>
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {selectedExercise && (
          <SpeechPractice 
            key={selectedExercise.id} // 🔥 FORCE REMOUNT WHEN LEVEL CHANGES
            exercise={selectedExercise}
            onClose={() => setSelectedExercise(null)}
            onComplete={(sessionData) => handleComplete(selectedExercise, sessionData)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default Training;
