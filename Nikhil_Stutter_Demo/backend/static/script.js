/* ============================================
   FluencyAI — Frontend Logic
   ============================================ */

let mediaRecorder;
let recordedChunks = [];
let timerInterval = null;
let recordingStartTime = null;
let currentStream = null;

// --- DOM ELEMENTS ---
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const preview = document.getElementById("preview");
const recordingDot = document.getElementById("recordingDot");
const timerOverlay = document.getElementById("timerOverlay");
const processingOverlay = document.getElementById("processingOverlay");
const stepReady = document.getElementById("stepReady");
const stepRecording = document.getElementById("stepRecording");
const stepProcessing = document.getElementById("stepProcessing");
const connectorOne = document.getElementById("connectorOne");
const connectorTwo = document.getElementById("connectorTwo");

if (preview) preview.style.display = "none";

// --- STEP MANAGEMENT ---
function setStep(step) {
    [stepReady, stepRecording, stepProcessing].forEach(s => {
        if (s) s.classList.remove("active", "completed");
    });
    [connectorOne, connectorTwo].forEach(c => {
        if (c) c.classList.remove("active");
    });

    if (step === "ready") {
        if (stepReady) stepReady.classList.add("active");
    } else if (step === "recording") {
        if (stepReady) stepReady.classList.add("completed");
        if (stepRecording) stepRecording.classList.add("active");
        if (connectorOne) connectorOne.classList.add("active");
    } else if (step === "processing") {
        if (stepReady) stepReady.classList.add("completed");
        if (stepRecording) stepRecording.classList.add("completed");
        if (stepProcessing) stepProcessing.classList.add("active");
        if (connectorOne) connectorOne.classList.add("active");
        if (connectorTwo) connectorTwo.classList.add("active");
    }
}

// --- TIMER ---
function startTimer() {
    recordingStartTime = Date.now();
    if (timerOverlay) timerOverlay.classList.add("active");

    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, "0");
        const secs = (elapsed % 60).toString().padStart(2, "0");
        if (timerOverlay) timerOverlay.textContent = `${mins}:${secs}`;
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
    if (timerOverlay) {
        timerOverlay.classList.remove("active");
        timerOverlay.textContent = "00:00";
    }
}

// --- STREAM CLEANUP ---
function stopStream() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    if (preview) {
        preview.srcObject = null;
        preview.style.display = "none"; // 🔥 HIDE AGAIN
    }
}

// --- PROCESSING OVERLAY ---
function showProcessing() {
    if (processingOverlay) processingOverlay.classList.add("active");
}

function hideProcessing() {
    if (processingOverlay) processingOverlay.classList.remove("active");
}

// --- START RECORDING ---
startBtn.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });

        currentStream = stream;
        preview.srcObject = stream;

        preview.style.display = "block";

        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) recordedChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            stopTimer();
            stopStream();
            if (recordingDot) recordingDot.classList.remove("active");

            setStep("processing");
            showProcessing();
            await uploadVideo();
        };

        mediaRecorder.start();

        startBtn.disabled = true;
        stopBtn.disabled = false;
        setStep("recording");
        startTimer();
        if (recordingDot) recordingDot.classList.add("active");

    } catch (err) {
        console.error("Camera/mic error:", err);
        alert("Could not access camera or microphone. Please allow permissions and try again.");
    }
});

// --- STOP RECORDING ---
stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

// --- UPLOAD ---
async function uploadVideo() {
    try {
        const blob = new Blob(recordedChunks, { type: "video/webm" });
        const formData = new FormData();
        formData.append("video", blob, "recording.webm");

        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Server error (${response.status})`);
        }

        const data = await response.json();

        if (!data.session_id) {
            throw new Error("No session ID returned");
        }

        localStorage.setItem("last_session", data.session_id);
        saveSessionToHistory(data.session_id, data.results);

        hideProcessing();
        window.location.href = `/result?session=${data.session_id}`;

    } catch (err) {
        console.error("Upload error:", err);
        hideProcessing();
        setStep("ready");
        alert(`Processing failed: ${err.message}\n\nPlease try again.`);
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// --- SESSION HISTORY ---
function saveSessionToHistory(sessionId, results) {
    try {
        let history = JSON.parse(localStorage.getItem("session_history") || "[]");

        history.unshift({
            session_id: sessionId,
            timestamp: Date.now(),
            fluency_score: results?.summary?.fluency_score ?? null,
            total_events: results?.summary?.total_events ?? 0,
            duration: results?.summary?.duration ?? 0
        });

        history = history.slice(0, 10);
        localStorage.setItem("session_history", JSON.stringify(history));
    } catch (e) {
        console.error("Failed to save session history:", e);
    }
}

function loadSessionHistory() {
    const container = document.getElementById("historyList");
    if (!container) return;

    try {
        const history = JSON.parse(localStorage.getItem("session_history") || "[]");

        if (history.length === 0) {
            container.innerHTML = `<div class="empty-state">No previous sessions yet.</div>`;
            return;
        }

        container.innerHTML = "";

        history.forEach(entry => {
            const date = new Date(entry.timestamp);
            const dateStr = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
            const timeStr = date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
            const score = entry.fluency_score !== null ? entry.fluency_score : "\u2014";
            const duration = entry.duration ? `${Math.round(entry.duration)}s` : "";
            const events = entry.total_events || 0;

            const item = document.createElement("a");
            item.className = "history-item";
            item.href = `/result?session=${entry.session_id}`;
            item.innerHTML = `
                <div class="history-score-badge" style="background:${getScoreColor(entry.fluency_score)}">${score}</div>
                <div class="history-info">
                    <div class="history-date">${dateStr} at ${timeStr}</div>
                    <div class="history-meta">${events} events \u00B7 ${duration}</div>
                </div>
                <span class="history-chevron">\u203A</span>
            `;
            container.appendChild(item);
        });
    } catch (e) {
        console.error("Failed to load history:", e);
    }
}

function getScoreColor(score) {
    if (score === null || score === undefined) return "#9BA5B1";
    if (score >= 80) return "#2BA882";
    if (score >= 60) return "#E8A838";
    if (score >= 40) return "#E8A838";
    return "#E05D5D";
}

// --- INIT ---
setStep("ready");
loadSessionHistory();