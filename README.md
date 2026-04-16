# 🎙️ VocaCare | AI-Powered Fluency Training System

<div align="center">
  <img src="frontend/public/favicon.svg" width="100" height="100" alt="VocaCare Logo">
  <h3>Clinical-Grade Speech Analysis with a Premium SaaS Aesthetic</h3>
  <p>
    <img src="https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue?style=for-the-badge&logo=react" alt="Frontend">
    <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=for-the-badge&logo=fastapi" alt="Backend">
    <img src="https://img.shields.io/badge/Database-MongoDB-darkgreen?style=for-the-badge&logo=mongodb" alt="Database">
  </p>
</div>

---

## 🌟 Overview
VocaCare is an advanced AI-driven platform designed to assist individuals with speech impediments, specifically stuttering. It combines real-time speech recognition, motion analysis, and clinical training exercises to provide a comprehensive fluency improvement experience.

## ✨ Key Features
- **💎 Premium UI/UX**: High-impact design with glassmorphism, smooth animations (Framer Motion), and adaptive dark/light themes.
- **🔐 Unified Auth**: Secure JWT-based authentication with a seamless interactive login/signup experience.
- **🏋️ Speech Training**: 5 clinical categories (SoundRep, WordRep, etc.) with 15 specialized exercises and real-time word-matching algorithms.
- **📊 Fluency Dashboard**: Dynamic statistics tracking including 24-hour streaks, weighted fluency scores, and historical progress charts.
- **🎥 Media Analysis**: 1080p recording with background ML inference for detecting stutter events and head movements.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Frontend** | React 19, Vite, TypeScript, Tailwind CSS 4, Framer Motion, Zustand |
| **Backend** | FastAPI (Python 3.14+), Uvicorn, Motor (Async MongoDB), JWT, Bcrypt |
| **Database** | MongoDB Atlas |
| **ML Engine** | MediaPipe, Web Speech API |

---

## 🚀 Getting Started

### 📋 Prerequisites
- **Node.js** (v18+)
- **Python** (v3.10+)
- **MongoDB Atlas** Account

### 🔧 Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/heyayushhh/Projexa-AI.git
cd Projexa-AI
```

#### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Update .env with your MongoDB credentials
```

#### 3. Frontend Setup
```bash
cd ../frontend
npm install
cp .env.example .env.local  # Update VITE_API_URL to http://127.0.0.1:8000/api/v1
```

### 🏃 Running the Project

**Start Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

**Start Frontend:**
```bash
cd frontend
npm run dev
```

---

## ☁️ Deployment

### **Backend (FastAPI)**
Recommended: [Render](https://render.com) or [Railway](https://railway.app)
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### **Frontend (React)**
Recommended: [Vercel](https://vercel.com) or [Netlify](https://netlify.com)
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Env Variable**: `VITE_API_URL` pointing to your deployed backend.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
<p align="center">Made with ❤️ for better communication.</p>
