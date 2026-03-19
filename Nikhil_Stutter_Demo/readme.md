# 🧠 Stutter Detection AI (Multimodal Speech Analysis)

## 🚀 Overview

This project is a **multimodal AI system** that detects speech stuttering using:

- 🎧 Audio analysis (Hugging Face model)
- 🎥 Visual analysis (MediaPipe + OpenCV)
- 📊 Timeline + insights + playback UI

The system records a user session, processes it locally, and provides:
- Stutter timestamps
- Stutter types
- Visual indicators (head movement)
- Insights about speech patterns

---

## 🎯 Features (Current)

### ✅ Recording System
- Webcam + microphone recording via browser
- Saves session locally
- Stores:
  - `video.webm`
  - `audio.wav`
  - `results.json`

---

### ✅ Audio Stutter Detection
Using:
- `HareemFatima/distilhubert-finetuned-stutterdetection`

Detects:
- fluent
- prolongation
- block
- sound repetition
- word repetition
- difficult speech
- interjection

---

### ✅ Visual Detection (Basic)
Using:
- MediaPipe Face Mesh
- OpenCV

Detects:
- head jerks (based on motion, velocity, acceleration)

⚠️ Note:
- Still heuristic-based
- Needs improvement for accuracy

---

### ✅ Processing Pipeline

1. Record session
2. Extract audio using FFmpeg
3. Run:
   - Audio model
   - Visual model
4. Merge results
5. Generate:
   - events
   - summary
   - insights
6. Save to `results.json`

---

### ✅ Result Dashboard UI

- 📊 Summary panel
- 🧠 Insights
- 🎥 Video playback
- 📍 Clickable timeline
- 🎨 Color-coded stutters

---

## 📁 Project Structure
# 🧠 Stutter Detection AI (Multimodal Speech Analysis)

## 🚀 Overview

This project is a **multimodal AI system** that detects speech stuttering using:

- 🎧 Audio analysis (Hugging Face model)
- 🎥 Visual analysis (MediaPipe + OpenCV)
- 📊 Timeline + insights + playback UI

The system records a user session, processes it locally, and provides:
- Stutter timestamps
- Stutter types
- Visual indicators (head movement)
- Insights about speech patterns

---

## 🎯 Features (Current)

### ✅ Recording System
- Webcam + microphone recording via browser
- Saves session locally
- Stores:
  - `video.webm`
  - `audio.wav`
  - `results.json`

---

### ✅ Audio Stutter Detection
Using:
- `HareemFatima/distilhubert-finetuned-stutterdetection`

Detects:
- fluent
- prolongation
- block
- sound repetition
- word repetition
- difficult speech
- interjection

---

### ✅ Visual Detection (Basic)
Using:
- MediaPipe Face Mesh
- OpenCV

Detects:
- head jerks (based on motion, velocity, acceleration)

⚠️ Note:
- Still heuristic-based
- Needs improvement for accuracy

---

### ✅ Processing Pipeline

1. Record session
2. Extract audio using FFmpeg
3. Run:
   - Audio model
   - Visual model
4. Merge results
5. Generate:
   - events
   - summary
   - insights
6. Save to `results.json`

---

### ✅ Result Dashboard UI

- 📊 Summary panel
- 🧠 Insights
- 🎥 Video playback
- 📍 Clickable timeline
- 🎨 Color-coded stutters

---

## 📁 Project Structure

```
backend/
│
├── app.py # Flask server
├── processor.py # Main pipeline
├── audio_model.py # Audio detection
├── visual_model.py # Visual detection
│
├── templates/
│ ├── index.html # Recording page
│ └── result.html # Results dashboard
│
├── static/
│ ├── script.js # Frontend logic
│ └── style.css # UI styling
│
recordings/
└── <session_id>/
├── video.webm
├── audio.wav
└── results.json
```

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Run backend
```bash
python app.py
```

### 3. Open UI
```
http://127.0.0.1:5000/app
```


### ⚠️ Known Limitations
1. Visual Detection

- Not very accurate yet
- Uses simple motion heuristics
- Needs head pose estimation (yaw/pitch/roll)

2. No Real-Time Processing

- Works only after recording ends

3. No Database

- Everything stored locally

4. No Authentication / User Profiles


### 🚀 NEXT STEPS (IMPORTANT)
1. 🔥 Priority 1 — Improve Visual Detection

- Use head pose estimation
- Detect lip tremor
- Reduce noise in detection

2. 🔥 Priority 2 — Fluency Score System

- Add scoring based on:
- frequency
- severity
- duration
- Display score in UI

3. 🔥 Priority 3 — Timeline Visualization

- Add graphical timeline (chart)
- Show stutters visually across time

4. 🔥 Priority 4 — Clip Extraction

- Allow playing only specific stutter segments

5. 🔥 Priority 5 — UI Improvements

- Better dashboard design
- Add icons, colors, animations

6. 🔥 Priority 6 — Real-Time Detection (Advanced)

- Live feedback while speaking
- Streaming audio + video inference

7. 🔥 Priority 7 — Backend Upgrade

- Convert to API-based architecture
- Add database (sessions history)

### 🧠 Future Vision

This can evolve into:
- 🗣️ Speech therapy assistant
- 📊 Personal fluency tracker
- 🤖 AI speech coach
- 🏥 Clinical support tool

### 💡 Key Insight

This project is not just:

"stutter detection"

It is:

Multimodal behavioral analysis using AI

### 👨‍💻 For Next AI / Developer

If you are continuing this project:

- DO NOT rewrite everything 
- Focus on improving:
- Visual detection accuracy
- Scoring system
- UI polish

Avoid:
- overcomplicating backend
- adding too many features at once

### ✅ Current Status

- MVP COMPLETE
- End-to-end working
- Ready for demo / portfolio