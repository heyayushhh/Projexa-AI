import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useThemeStore } from '../store/themeStore';

import { authApi } from '../services/api';

interface AuthPageProps {
  onAuthSuccess: () => void;
}

const AuthPage = ({ onAuthSuccess }: AuthPageProps) => {
  const [isLogin, setIsLogin] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { isDark, login: setStoreLogin } = useThemeStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    const formData = new FormData(e.currentTarget as HTMLFormElement);
    const data = Object.fromEntries(formData.entries());
    const asString = (value: FormDataEntryValue | undefined) => (typeof value === 'string' ? value : '');

    try {
      if (isLogin) {
        const res = await authApi.login({
          username: asString(data.usernameOrEmail as FormDataEntryValue | undefined),
          password: asString(data.password as FormDataEntryValue | undefined),
        });
        setStoreLogin(res.access_token, res.user);
      } else {
        await authApi.register({
          name: asString(data.fullName as FormDataEntryValue | undefined),
          username: asString(data.username as FormDataEntryValue | undefined),
          description: asString(data.description as FormDataEntryValue | undefined),
          email: asString(data.email as FormDataEntryValue | undefined),
          password: asString(data.password as FormDataEntryValue | undefined),
        });
        // Auto-login after register
        const res = await authApi.login({
          username: asString(data.username as FormDataEntryValue | undefined),
          password: asString(data.password as FormDataEntryValue | undefined),
        });
        setStoreLogin(res.access_token, res.user);
      }
      onAuthSuccess();
    } catch (err: unknown) {
      const maybeDetail =
        typeof err === 'object' && err !== null
          ? (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail
          : null;
      setError(typeof maybeDetail === 'string' ? maybeDetail : 'Authentication failed');
    } finally {
      setIsLoading(false);
    }
  };

  const labelClasses = `text-sm font-medium tracking-wide transition-colors duration-200 ${
    isDark 
      ? 'text-gray-300 group-focus-within:text-red-400' 
      : 'text-gray-700 group-focus-within:text-red-500'
  }`;

  return (
    <div className="flex flex-col items-center justify-center min-h-[85vh] px-4 sm:px-6 md:px-8 py-10 sm:py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ 
          opacity: 1, 
          y: [0, -10, 0],
        }}
        transition={{ 
          opacity: { duration: 0.6 },
          y: { 
            duration: 5, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }
        }}
        className={`backdrop-blur-2xl p-6 sm:p-8 rounded-[2rem] border shadow-[0_25px_50px_-12px_rgba(0,0,0,0.25)] w-full max-w-md transition-all duration-500 ${
          isDark 
            ? 'bg-black/40 border-white/10 shadow-black/40' 
            : 'bg-white/70 border-black/5 shadow-black/10'
        }`}
      >
        {/* Auth Toggle Switch - FIXED wrapping and stability */}
        <div className={`flex justify-center mb-5 p-1 rounded-2xl w-full max-w-[260px] mx-auto transition-colors duration-500 ${
          isDark ? 'bg-white/5' : 'bg-black/5'
        }`}>
          <div className="relative flex items-center w-full gap-2">
            <motion.div
              className={`absolute h-full rounded-xl transition-all duration-300 ${
                isDark ? 'bg-gray-800' : 'bg-white shadow-sm'
              }`}
              initial={false}
              animate={{ 
                x: isLogin ? 0 : '100%',
                left: isLogin ? 0 : -8, 
              }}
              style={{ width: 'calc(50% + 4px)' }}
            />
            <button 
              onClick={() => setIsLogin(true)}
              className={`relative z-10 flex-1 py-2 rounded-xl transition-colors duration-300 font-bold whitespace-nowrap text-center ${
                isLogin 
                  ? 'text-[#FF2E2E]' 
                  : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              Login
            </button>
            <button 
              onClick={() => setIsLogin(false)}
              className={`relative z-10 flex-1 py-2 rounded-xl transition-colors duration-300 font-bold whitespace-nowrap text-center ${
                !isLogin 
                  ? 'text-[#FF2E2E]' 
                  : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              Sign Up
            </button>
          </div>
        </div>

        <h2 className={`text-3xl font-bold mb-5 text-center tracking-tight transition-colors duration-500 ${
          isDark ? 'text-white' : 'text-[#111]'
        }`}>
          {isLogin ? 'Welcome Back' : 'Create Account'}
        </h2>
        
        <form className="space-y-6 mb-5" onSubmit={handleSubmit}>
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-500 p-3 rounded-xl text-center text-sm font-medium">
              {error}
            </div>
          )}
          <AnimatePresence mode="wait">
            {isLogin ? (
              <motion.div
                key="login-fields"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="space-y-5"
              >
                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Username or Email
                  </label>
                  <input 
                    name="usernameOrEmail"
                    type="text" 
                    placeholder="Enter username or email" 
                    className={`input input-bordered h-12 bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                </div>
                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Password
                  </label>
                  <input 
                    name="password"
                    type="password" 
                    placeholder="••••••••" 
                    className={`input input-bordered h-12 bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="signup-fields"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="space-y-5"
              >
                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Full Name
                  </label>
                  <input 
                    name="fullName"
                    type="text" 
                    placeholder="John Doe" 
                    className={`input input-bordered h-12 bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                </div>

                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Username
                  </label>
                  <input 
                    name="username"
                    type="text" 
                    placeholder="johndoe" 
                    className={`input input-bordered h-12 bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                </div>

                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Description
                  </label>
                  <textarea 
                    name="description"
                    placeholder="Tell us about yourself..." 
                    className={`textarea textarea-bordered bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl resize-none min-h-[90px] transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                  <span className="text-[10px] text-gray-500 pl-1 font-medium">Max 20 words</span>
                </div>

                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Email Address
                  </label>
                  <input 
                    name="email"
                    type="email" 
                    placeholder="name@example.com" 
                    className={`input input-bordered h-12 bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                </div>
                
                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Password
                  </label>
                  <input 
                    name="password"
                    type="password" 
                    placeholder="••••••••" 
                    className={`input input-bordered h-12 bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                </div>

                <div className="form-control group flex flex-col gap-1.5">
                  <label className={labelClasses}>
                    Confirm Password
                  </label>
                  <input 
                    name="confirmPassword"
                    type="password" 
                    placeholder="••••••••" 
                    className={`input input-bordered h-12 bg-white/5 dark:bg-black/20 border-black/10 dark:border-white/10 rounded-xl transition-all focus:ring-2 focus:ring-red-500/40 focus:border-red-500/40 outline-none w-full ${
                      isDark ? 'text-white' : 'text-[#111]'
                    }`}
                    required
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          <div className="pt-2.5">
            <motion.button
              whileHover={{ scale: 1.02, boxShadow: '0 10px 15px -3px rgba(255,46,46,0.2)' }}
              whileTap={{ scale: 0.98 }}
              className="w-full bg-[#FF2E2E] hover:bg-[#E02929] text-white font-bold py-3.5 rounded-xl shadow-lg transition-all duration-300 border-none text-base cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? (
                <span className="loading loading-spinner loading-sm"></span>
              ) : (
                isLogin ? 'Login to VocaCare' : 'Join VocaCare Now'
              )}
            </motion.button>
          </div>
        </form>
        
        <p className={`text-center transition-colors duration-500 font-medium text-sm ${isDark ? 'text-gray-400' : 'text-[#444]'}`}>
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <button 
            onClick={() => setIsLogin(!isLogin)}
            className="ml-2 text-red-500 hover:text-red-600 font-bold cursor-pointer underline underline-offset-4 decoration-2"
          >
            {isLogin ? 'Sign up' : 'Login'}
          </button>
        </p>
      </motion.div>
    </div>
  );
};

export default AuthPage;
