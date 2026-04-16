import { motion } from 'framer-motion';

interface LoginProps {
  onLogin: () => void;
  onSwitch: () => void;
}

const Login = ({ onLogin, onSwitch }: LoginProps) => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh]">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="backdrop-blur-xl bg-white/10 dark:bg-black/30 p-10 rounded-2xl border border-white/20 shadow-2xl w-full max-w-md"
      >
        <h2 className="text-3xl font-bold mb-8 text-center text-gray-900 dark:text-white">Welcome Back</h2>
        
        <form className="space-y-6" onSubmit={(e) => { e.preventDefault(); onLogin(); }}>
          <div className="form-control">
            <label className="label">
              <span className="label-text text-gray-700 dark:text-gray-300">Email Address</span>
            </label>
            <input 
              type="email" 
              placeholder="name@example.com" 
              className="input input-bordered bg-white/5 dark:bg-black/20 focus:ring-2 focus:ring-red-500 transition-all border-white/20"
              required
            />
          </div>
          
          <div className="form-control">
            <label className="label">
              <span className="label-text text-gray-700 dark:text-gray-300">Password</span>
            </label>
            <input 
              type="password" 
              placeholder="••••••••" 
              className="input input-bordered bg-white/5 dark:bg-black/20 focus:ring-2 focus:ring-red-500 transition-all border-white/20"
              required
            />
          </div>
          
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full bg-[#FF2E2E] hover:bg-[#E02929] text-white font-bold py-3 rounded-xl shadow-lg transition-all duration-300 mt-4"
            type="submit"
          >
            Login
          </motion.button>
        </form>
        
        <p className="mt-8 text-center text-gray-700 dark:text-gray-400">
          Don't have an account? 
          <button 
            onClick={onSwitch}
            className="ml-2 text-red-500 hover:text-red-600 font-semibold cursor-pointer"
          >
            Sign up
          </button>
        </p>
      </motion.div>
    </div>
  );
};

export default Login;
