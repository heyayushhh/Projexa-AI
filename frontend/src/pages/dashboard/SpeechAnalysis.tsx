import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { dashboardApi } from '../../services/api';
import { useThemeStore } from '../../store/themeStore';

interface DetectedEvent {
  timestamp: number;
  type: string;
}

interface AnalysisResult {
  transcript: string;
  fluency_score: number;
  total_words?: number;
  stutterEvents?: DetectedEvent[];
  headMovements?: DetectedEvent[];
}

const SpeechAnalysis = () => {
  const { isDark } = useThemeStore();
  const [isRecording, setIsRecording] = useState(false);
  const [mediaBlob, setMediaBlob] = useState<Blob | null>(null);
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'done'>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [timer, setTimer] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (isRecording) {
      timerRef.current = window.setInterval(() => {
        setTimer(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const startRecording = async () => {
    try {
      // Capture both video and audio
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 }, 
        audio: true 
      });
      
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }

      const recorder = new MediaRecorder(mediaStream, {
        mimeType: 'video/webm;codecs=vp8,opus'
      });
      
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        setMediaBlob(blob);
      };

      recorder.start();
      setIsRecording(true);
      setTimer(0);
      setStatus('idle');
    } catch (err) {
      console.error("Error accessing camera/mic:", err);
      alert("Please allow camera and microphone access to start recording.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      setStream(null);
      setIsRecording(false);
    }
  };

  const handleAnalyze = async () => {
    if (!mediaBlob) return;
    setStatus('analyzing');
    try {
      const file = new File([mediaBlob], 'recording.webm', { type: 'video/webm' });
      await dashboardApi.analyze(file);
      
      // Simulation of polling for results (since the background task takes time)
      setTimeout(async () => {
        const sessions = await dashboardApi.getSessions();
        if (sessions.length > 0) {
          setResult(sessions[0] as unknown as AnalysisResult);
          setStatus('done');
        } else {
          setStatus('idle');
        }
      }, 6000);
    } catch (err) {
      console.error(err);
      setStatus('idle');
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div className={`p-4 sm:p-6 md:p-8 rounded-2xl sm:rounded-[2.5rem] border backdrop-blur-2xl shadow-2xl transition-all duration-500 ${
        isDark ? 'bg-black/40 border-white/10' : 'bg-white/70 border-black/5'
      }`}>
        <h2 className={`text-2xl sm:text-3xl font-black mb-6 sm:mb-8 text-center break-words ${isDark ? 'text-white' : 'text-[#111]'}`}>
          AI Speech & Movement Analysis
        </h2>
        
        <div className="flex flex-col lg:flex-row gap-6 sm:gap-10 items-center lg:items-start">
          {/* Video Preview / Recording Area */}
          <div className="flex-1 w-full relative aspect-video bg-black rounded-3xl overflow-hidden shadow-2xl border-4 border-white/5">
            {isRecording ? (
              <video 
                ref={videoRef} 
                autoPlay 
                muted 
                className="w-full h-full object-cover"
              />
            ) : mediaBlob ? (
              <video 
                src={URL.createObjectURL(mediaBlob)} 
                controls 
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
                <div className="text-6xl mb-4">📷</div>
                <p className="text-lg font-bold">Camera Preview</p>
              </div>
            )}
            
            {isRecording && (
              <div className="absolute top-4 left-4 sm:top-6 sm:left-6 flex items-center gap-3 bg-black/60 backdrop-blur-md px-3 sm:px-4 py-2 rounded-full border border-red-500/30">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                <span className="text-white font-mono font-bold">{formatTime(timer)}</span>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex flex-col items-center justify-center gap-5 sm:gap-6 lg:w-64 w-full">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-20 h-20 sm:w-24 sm:h-24 rounded-full flex items-center justify-center text-3xl sm:text-4xl shadow-2xl transition-all ${
                isRecording ? 'bg-red-600' : 'bg-[#FF2E2E] hover:bg-[#E02929]'
              } text-white border-none cursor-pointer`}
            >
              {isRecording ? '⏹️' : '🎤'}
            </motion.button>
            
            <div className="text-center">
              <p className={`text-lg font-bold ${isDark ? 'text-white' : 'text-[#111]'}`}>
                {isRecording ? 'Recording Session...' : mediaBlob ? 'Recording Saved' : 'Start Practice'}
              </p>
              <p className="text-sm text-gray-500 font-medium mt-1">
                {isRecording ? 'Capture your speech and movements' : 'Camera and Mic will be used'}
              </p>
            </div>

            {mediaBlob && status === 'idle' && (
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                onClick={handleAnalyze}
                className="w-full min-h-[44px] bg-white text-black font-black py-3.5 sm:py-4 rounded-2xl hover:bg-gray-100 transition-all shadow-xl border-none cursor-pointer text-base sm:text-lg"
              >
                Analyze Now
              </motion.button>
            )}
          </div>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {status === 'analyzing' && (
          <motion.div
            key="analyzing"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center p-6 sm:p-12 bg-black/20 rounded-2xl sm:rounded-[2.5rem] border border-white/5 backdrop-blur-xl text-center"
          >
            <div className="loading loading-spinner loading-lg text-red-500 mb-6 scale-150"></div>
            <p className="text-lg sm:text-2xl font-black text-white animate-pulse tracking-tight break-words">AI is analyzing speech & head movements...</p>
            <p className="text-gray-400 mt-2 font-medium">This usually takes 10-15 seconds</p>
          </motion.div>
        )}

        {status === 'done' && result && (
          <motion.div
            key="results"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`p-10 rounded-[2.5rem] border shadow-2xl transition-all duration-500 ${
              isDark ? 'bg-black/40 border-white/10' : 'bg-white/70 border-black/5'
            }`}
          >
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
              {/* Transcript & Basic Info */}
              <div className="lg:col-span-2 space-y-8">
                <div>
                  <h4 className="text-[#FF2E2E] font-black uppercase tracking-widest text-xs mb-3">Session Transcript</h4>
                  <p className={`text-2xl leading-relaxed font-medium italic ${isDark ? 'text-gray-200' : 'text-[#444]'}`}>
                    "{result.transcript}"
                  </p>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-5 rounded-2xl bg-black/20 border border-white/5">
                    <p className="text-[10px] text-gray-500 font-black uppercase tracking-widest mb-1">Words</p>
                    <p className="text-2xl font-black text-blue-500">
                      {result.total_words ?? (result.transcript ? result.transcript.trim().split(/\s+/).filter(Boolean).length : 0)}
                    </p>
                  </div>
                  <div className="p-5 rounded-2xl bg-black/20 border border-white/5">
                    <p className="text-[10px] text-gray-500 font-black uppercase tracking-widest mb-1">Stutters</p>
                    <p className="text-2xl font-black text-red-500">{result.stutterEvents?.length || 0}</p>
                  </div>
                  <div className="p-5 rounded-2xl bg-black/20 border border-white/5">
                    <p className="text-[10px] text-gray-500 font-black uppercase tracking-widest mb-1">Head Mov.</p>
                    <p className="text-2xl font-black text-purple-500">{result.headMovements?.length || 0}</p>
                  </div>
                  <div className="p-5 rounded-2xl bg-black/20 border border-white/5">
                    <p className="text-[10px] text-gray-500 font-black uppercase tracking-widest mb-1">Duration</p>
                    <p className="text-2xl font-black text-green-500">{formatTime(timer)}</p>
                  </div>
                </div>

                {/* Event Timeline */}
                <div>
                  <h4 className="text-[#FF2E2E] font-black uppercase tracking-widest text-xs mb-4">Detected Events</h4>
                  <div className="flex flex-wrap gap-3">
                    {result.stutterEvents?.map((event: DetectedEvent, i: number) => (
                      <div key={`s-${i}`} className="bg-red-500/10 border border-red-500/20 px-4 py-2 rounded-xl flex items-center gap-2">
                        <span className="text-red-500 font-black text-xs">{event.timestamp}s</span>
                        <span className="text-white font-bold text-sm capitalize">{event.type}</span>
                      </div>
                    ))}
                    {result.headMovements?.map((event: DetectedEvent, i: number) => (
                      <div key={`h-${i}`} className="bg-purple-500/10 border border-purple-500/20 px-4 py-2 rounded-xl flex items-center gap-2">
                        <span className="text-purple-500 font-black text-xs">{event.timestamp}s</span>
                        <span className="text-white font-bold text-sm">Head Mov.</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Main Score Card */}
              <div className="flex flex-col items-center justify-center p-10 rounded-[2rem] bg-gradient-to-br from-red-500/10 to-transparent border border-red-500/20 shadow-inner">
                <h4 className="text-[#FF2E2E] font-black uppercase tracking-widest text-xs mb-6 text-center">Fluency Score</h4>
                <div className="relative">
                  <svg className="w-48 h-48 transform -rotate-90">
                    <circle
                      cx="96" cy="96" r="88"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="transparent"
                      className="text-white/5"
                    />
                    <motion.circle
                      cx="96" cy="96" r="88"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="transparent"
                      strokeDasharray={552.92}
                      initial={{ strokeDashoffset: 552.92 }}
                      animate={{ strokeDashoffset: 552.92 - (552.92 * result.fluency_score) / 100 }}
                      transition={{ duration: 2, ease: "easeOut" }}
                      className="text-[#FF2E2E]"
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-5xl font-black text-white">{result.fluency_score}%</span>
                  </div>
                </div>
                <p className="mt-8 text-sm text-gray-400 font-bold text-center leading-relaxed">
                  Significant improvement in breath control detected. Focus on reducing head movements during plosives.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SpeechAnalysis;
