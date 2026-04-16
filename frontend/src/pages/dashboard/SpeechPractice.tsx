import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Exercise } from './exerciseData';
import { getExerciseTypeLabel } from './exerciseData';

interface SpeechPracticeProps {
  exercise: Exercise;
  onComplete: (result: {
    spokenText: string;
    expectedText: string;
    fluencyScore: number;
    isCorrect: boolean;
    wordCount: number;
    errorCount: number;
  }) => Promise<void>; // 🔥 Ensure it's treated as a Promise
  onClose: () => void;
}

interface SpeechRecognitionAlternative {
  transcript: string;
}

interface SpeechRecognitionResult {
  0: SpeechRecognitionAlternative;
  isFinal: boolean;
}

interface SpeechRecognitionEvent {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResult>;
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

interface ISpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onstart: (() => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionConstructor = new () => ISpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

const SpeechPractice: React.FC<SpeechPracticeProps> = ({ exercise, onComplete, onClose }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const transcriptRef = useRef(''); // 🔥 USE REF TO AVOID RACE CONDITIONS
  const [recognition, setRecognition] = useState<ISpeechRecognition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false); // 🔥 LOADING STATE FOR API
  const [result, setResult] = useState<{
    score: number;
    isCorrect: boolean;
    wordCount: number;
    errorCount: number;
    grade: string;
    highlights: { word: string; status: 'correct' | 'wrong' | 'missing' }[];
  } | null>(null);

  // Similarity helper (Levenshtein Distance)
  const getSimilarity = useCallback((s1: string, s2: string) => {
    const len1 = s1.length;
    const len2 = s2.length;
    if (len1 === 0 || len2 === 0) return 0;
    
    const matrix = Array.from({ length: len1 + 1 }, () => Array(len2 + 1).fill(0));
    for (let i = 0; i <= len1; i++) matrix[i][0] = i;
    for (let j = 0; j <= len2; j++) matrix[0][j] = j;
    
    for (let i = 1; i <= len1; i++) {
      for (let j = 1; j <= len2; j++) {
        const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
        matrix[i][j] = Math.min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost);
      }
    }
    return 1 - matrix[len1][len2] / Math.max(len1, len2);
  }, []);

  const calculateAdvancedScore = useCallback((expected: string, spoken: string) => {
    const clean = (str: string) => 
      str.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(w => w.length > 0);

    const exp = clean(expected);
    const spk = clean(spoken);

    let i = 0; // expected pointer
    let j = 0; // spoken pointer
    let matches = 0;
    const result: { word: string; status: 'correct' | 'wrong' | 'missing' }[] = [];

    // If spoken is empty, everything is missing
    if (spk.length === 0) {
      return {
        score: 0,
        isCorrect: false,
        wordCount: 0,
        errorCount: exp.length,
        grade: 'Try Again',
        highlights: exp.map(w => ({ word: w, status: 'missing' as const }))
      };
    }

    while (i < exp.length) {
      const expectedWord = exp[i];
      const spokenWord = j < spk.length ? spk[j] : null;
      const nextSpokenWord = j + 1 < spk.length ? spk[j + 1] : null;

      // 1. Perfect Match
      if (spokenWord && expectedWord === spokenWord) {
        result.push({ word: expectedWord, status: "correct" });
        matches += 1;
        i++; j++;
      } 
      // 2. Fuzzy Match (Misspelled / Similar)
      else if (spokenWord && getSimilarity(expectedWord, spokenWord) > 0.5) { // 0.5 is very forgiving
        result.push({ word: expectedWord, status: "wrong" });
        matches += 0.9; // 🔥 High credit for misspelled but recognized words
        i++; j++;
      }
      // 3. Extra word in spoken (skip current spoken and look ahead)
      else if (nextSpokenWord && (expectedWord === nextSpokenWord || getSimilarity(expectedWord, nextSpokenWord) > 0.5)) {
        if (expectedWord === nextSpokenWord) {
          result.push({ word: expectedWord, status: "correct" });
          matches += 1;
        } else {
          result.push({ word: expectedWord, status: "wrong" });
          matches += 0.9;
        }
        i++; j += 2; // skip the extra word
      }
      // 4. Word skipped entirely
      else {
        result.push({ word: expectedWord, status: "missing" });
        i++;
      }
    }

    const score = Math.round((matches / exp.length) * 100);
    const cappedScore = Math.min(100, Math.max(0, score));

    let grade = 'Try Again';
    if (cappedScore >= 90) grade = 'Perfect';
    else if (cappedScore >= 75) grade = 'Great Job';
    else if (cappedScore >= 60) grade = 'Good Effort';
    else if (cappedScore >= 40) grade = 'Needs Improvement';

    return {
      score: cappedScore,
      isCorrect: cappedScore >= 70,
      wordCount: spk.length,
      errorCount: exp.length - Math.floor(matches),
      grade,
      highlights: result
    };
  }, [getSimilarity]);

  const startRecording = () => {
    if (recognition) {
      transcriptRef.current = ''; // 🔥 RESET REF
      setTranscript('');
      setResult(null);
      setError(null);
      try {
        recognition.start();
      } catch (e) {
        console.error("Failed to start recognition", e);
        setError("Speech recognition is already running or failed to start.");
      }
    }
  };

  const stopRecording = () => {
    if (recognition) {
      recognition.stop();
      setIsRecording(false);
    }
  };

  const handleProcessResult = useCallback(() => {
    const finalSpeech = transcriptRef.current; // 🔥 ALWAYS USE THE LATEST REF
    if (!finalSpeech) return;
    
    const stats = calculateAdvancedScore(exercise.sentence, finalSpeech);
    setResult(stats);
  }, [calculateAdvancedScore, exercise.sentence]);

  useEffect(() => {
    if (!SpeechRecognition) {
      setError("Web Speech API is not supported in this browser. Please use Chrome or Edge.");
      return;
    }

    const reco = new SpeechRecognition();
    reco.continuous = true;
    reco.interimResults = false;
    reco.lang = 'en-US';

    reco.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      if (finalTranscript) {
        const newTranscript = (transcriptRef.current + ' ' + finalTranscript).trim();
        transcriptRef.current = newTranscript;
        setTranscript(newTranscript);
      }
    };

    reco.onstart = () => {
      setIsRecording(true);
      setError(null);
    };

    reco.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === 'not-allowed') {
        setError("Microphone permission denied. Please enable mic access.");
      } else if (event.error === 'no-speech') {
        setError("No speech detected. Please try again.");
      } else {
        setError(`Error: ${event.error}`);
      }
      setIsRecording(false);
    };

    reco.onend = () => {
      setIsRecording(false);
      setTimeout(() => {
        handleProcessResult();
      }, 300);
    };

    setRecognition(reco);
    return () => {
      reco.onresult = null;
      reco.onerror = null;
      reco.onend = null;
      reco.onstart = null;
      reco.stop();
    };
  }, [handleProcessResult]);

  const handleFinish = async () => {
    if (result && !isSaving) {
      setIsSaving(true);
      setError(null); // Reset any previous errors
      try {
        await onComplete({
          spokenText: transcriptRef.current,
          expectedText: exercise.sentence,
          fluencyScore: result.score,
          isCorrect: result.isCorrect,
          wordCount: result.wordCount,
          errorCount: result.errorCount
        });
      } catch (e) {
        console.error("Save failed:", e);
        setError("Failed to save session. Please check your connection and try again.");
      } finally {
        setIsSaving(false);
      }
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-bg-primary/40 backdrop-blur-xl"
    >
      <motion.div 
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        className="w-full max-w-2xl max-h-[90vh] overflow-y-auto p-5 sm:p-8 md:p-10 rounded-2xl sm:rounded-[3rem] shadow-premium border border-border-subtle bg-card-bg transition-colors duration-300"
      >
        <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-4 mb-8 sm:mb-10">
          <div>
            <div className={`inline-flex px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${
              exercise.difficulty === 'Easy' ? 'bg-green-500/10 text-green-500' :
              exercise.difficulty === 'Medium' ? 'bg-blue-500/10 text-blue-500' :
              'bg-purple-500/10 text-purple-500'
            }`}>
              {getExerciseTypeLabel(exercise.type)} • {exercise.difficulty}
            </div>
            <h2 className="text-2xl sm:text-3xl font-black mt-3 text-text-primary tracking-tight break-words">
              Speech Practice
            </h2>
          </div>
          <button 
            onClick={onClose}
            className="min-h-[44px] min-w-[44px] p-3 rounded-2xl hover:bg-bg-primary text-text-secondary hover:text-text-primary transition-all duration-300 self-start sm:self-auto"
          >
            <span className="text-xl">✕</span>
          </button>
        </div>

        <div className="p-5 sm:p-8 md:p-10 rounded-2xl sm:rounded-[2.5rem] mb-8 sm:mb-10 border-2 border-dashed border-accent/20 bg-accent/5 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1 h-full bg-accent opacity-20"></div>
          <p className="text-[10px] uppercase tracking-[0.2em] font-black mb-4 text-accent/60">
            Read aloud clearly
          </p>
          <p className="text-xl sm:text-3xl md:text-4xl font-black leading-tight text-text-primary tracking-tight break-words">
            "{exercise.sentence}"
          </p>

          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-4 auto-rows-fr">
            <div className="p-4 rounded-2xl bg-bg-primary/40 border border-border-subtle h-full flex flex-col">
              <p className="text-[10px] font-black uppercase tracking-widest text-text-secondary mb-2">Target Focus</p>
              <p className="text-sm sm:text-base font-bold text-text-primary break-words flex-1">{exercise.targetFocus}</p>
            </div>
            <div className="p-4 rounded-2xl bg-bg-primary/40 border border-border-subtle h-full flex flex-col">
              <p className="text-[10px] font-black uppercase tracking-widest text-text-secondary mb-2">Technique</p>
              <p className="text-sm sm:text-base font-bold text-text-primary break-words flex-1">{exercise.technique}</p>
            </div>
            <div className="p-4 rounded-2xl bg-bg-primary/40 border border-border-subtle h-full flex flex-col">
              <p className="text-[10px] font-black uppercase tracking-widest text-text-secondary mb-2">Pacing</p>
              <p className="text-sm sm:text-base font-bold text-text-primary break-words flex-1">{exercise.pacing}</p>
            </div>
            <div className="p-4 rounded-2xl bg-bg-primary/40 border border-border-subtle h-full flex flex-col">
              <p className="text-[10px] font-black uppercase tracking-widest text-text-secondary mb-2">Tip</p>
              <p className="text-sm sm:text-base font-bold text-text-primary break-words flex-1">{exercise.tip}</p>
            </div>
          </div>
        </div>

        {error && (
          <div className="p-5 mb-8 rounded-2xl bg-orange-500/10 border border-orange-500/20 text-orange-600 text-sm font-bold flex items-center gap-3 animate-shake">
            <span className="text-lg">⚠️</span> {error}
          </div>
        )}

        {!result ? (
          <div className="flex flex-col items-center gap-8 sm:gap-10">
            <AnimatePresence mode="wait">
              {isRecording && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex flex-col items-center w-full"
                >
                  <div className="flex gap-1.5 mb-8 items-center h-16">
                    {[1, 2, 3, 4, 5, 6, 7].map(i => (
                      <motion.div
                        key={i}
                        animate={{ 
                          height: [15, 60, 15],
                          backgroundColor: ['#38B2AC', '#4FD1C5', '#38B2AC']
                        }}
                        transition={{ duration: 1, repeat: Infinity, delay: i * 0.1 }}
                        className="w-2 rounded-full shadow-[0_0_20px_rgba(56,178,172,0.3)]"
                      />
                    ))}
                  </div>
                  <div className="text-center space-y-4 w-full">
                    <p className="text-accent font-black uppercase tracking-[0.3em] text-sm animate-pulse">Recording...</p>
                    <div className="p-6 rounded-2xl bg-bg-primary/50 border border-border-subtle min-h-[80px] flex items-center justify-center">
                      <p className="text-base sm:text-xl font-bold text-text-primary italic opacity-80 break-words">
                        {transcript || "Speak now..."}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="relative">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="w-24 h-24 sm:w-32 sm:h-32 rounded-[2.25rem] sm:rounded-[3rem] bg-accent flex items-center justify-center shadow-[0_20px_50px_rgba(56,178,172,0.35)] sm:shadow-[0_20px_50px_rgba(56,178,172,0.4)] hover:scale-110 active:scale-95 transition-all duration-500 group"
                >
                  <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-2xl bg-white shadow-inner group-hover:rotate-90 transition-transform duration-500" />
                  <div className="absolute -bottom-9 sm:-bottom-10 left-1/2 -translate-x-1/2 whitespace-nowrap text-[10px] font-black uppercase tracking-widest text-text-secondary group-hover:text-accent transition-colors">
                    Click to Start
                  </div>
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="w-24 h-24 sm:w-32 sm:h-32 rounded-[2.25rem] sm:rounded-[3rem] bg-text-primary flex items-center justify-center shadow-premium hover:scale-110 active:scale-95 transition-all duration-500 group"
                >
                  <div className="w-10 h-10 bg-white rounded-xl group-hover:scale-75 transition-transform duration-500" />
                  <div className="absolute -bottom-9 sm:-bottom-10 left-1/2 -translate-x-1/2 whitespace-nowrap text-[10px] font-black uppercase tracking-widest text-accent animate-pulse">
                    Stop Now
                  </div>
                </button>
              )}
            </div>
          </div>
        ) : (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-10"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
              <div className="p-5 sm:p-8 rounded-2xl sm:rounded-[2.5rem] border border-border-subtle bg-bg-primary/30 group hover:bg-bg-primary transition-colors duration-500">
                <p className="text-[10px] font-black uppercase tracking-widest text-text-secondary mb-3">Accuracy</p>
                <div className="flex items-baseline gap-1">
                  <p className={`text-5xl font-black tracking-tighter ${
                    result.score >= 90 ? 'text-accent' : 
                    result.score >= 70 ? 'text-blue-500' : 
                    result.score >= 40 ? 'text-orange-500' : 'text-red-500'
                  }`}>
                    {result.score}%
                  </p>
                </div>
              </div>
              <div className="p-8 rounded-[2.5rem] border border-border-subtle bg-bg-primary/30 group hover:bg-bg-primary transition-colors duration-500">
                <p className="text-[10px] font-black uppercase tracking-widest text-text-secondary mb-3">Feedback</p>
                <p className={`text-3xl font-black uppercase tracking-tight leading-none ${
                  result.score >= 90 ? 'text-accent' : 
                  result.score >= 70 ? 'text-blue-500' : 
                  result.score >= 40 ? 'text-orange-500' : 'text-red-500'
                }`}>
                  {result.grade}
                </p>
              </div>
            </div>

            <div className="p-8 rounded-[2.5rem] border border-border-subtle bg-card-bg shadow-soft">
              <div className="flex justify-between items-center mb-6">
                <p className="text-[10px] font-black uppercase tracking-widest text-text-secondary">Word Analysis</p>
                <div className="flex gap-4">
                  <LegendItem color="bg-accent" label="Match" />
                  <LegendItem color="bg-orange-400" label="Partial" />
                  <LegendItem color="bg-border-subtle" label="Missed" />
                </div>
              </div>
              
              <div className="flex flex-wrap gap-x-3 gap-y-4 mb-8">
                {result.highlights.map((h, i) => (
                  <span 
                    key={i} 
                    className={`text-2xl font-black px-3 py-1 rounded-2xl transition-all duration-300 ${
                      h.status === 'correct' ? 'text-accent bg-accent/10' :
                      h.status === 'wrong' ? 'text-orange-500 bg-orange-500/10' :
                      'text-text-secondary bg-bg-primary opacity-40'
                    }`}
                  >
                    {h.word}
                  </span>
                ))}
              </div>

              <div className="pt-6 border-t border-border-subtle">
                <p className="text-[9px] uppercase font-black text-text-secondary/60 mb-3 tracking-widest">Recognized Audio:</p>
                <p className="text-lg font-bold text-text-primary italic leading-relaxed opacity-80">
                  "{transcript || "(Silence)"}"
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <button 
                onClick={() => {
                  setResult(null);
                  setTranscript('');
                  transcriptRef.current = '';
                }}
                className="flex-1 py-5 font-black uppercase tracking-widest text-[10px] rounded-2xl border border-border-subtle text-text-secondary hover:bg-bg-primary hover:text-text-primary transition-all duration-300"
              >
                Retry Level
              </button>
              <button 
                onClick={handleFinish}
                disabled={isSaving}
                className="flex-1 py-5 font-black uppercase tracking-widest text-[10px] rounded-2xl bg-accent text-white shadow-premium hover:bg-accent-hover hover:scale-[1.02] active:scale-[0.98] transition-all duration-300 disabled:opacity-50"
              >
                {isSaving ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                    Saving...
                  </span>
                ) : "Save & Continue"}
              </button>
            </div>
          </motion.div>
        )}
      </motion.div>
    </motion.div>
  );
};

const LegendItem = ({ color, label }: { color: string, label: string }) => (
  <div className="flex items-center gap-1.5">
    <div className={`w-2 h-2 rounded-full ${color}`}></div>
    <span className="text-[9px] font-black text-text-secondary uppercase tracking-tighter">{label}</span>
  </div>
);

export default SpeechPractice;
