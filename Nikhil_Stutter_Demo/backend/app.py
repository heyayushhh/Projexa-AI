from flask import Flask, request, jsonify, render_template, send_file
import os
import uuid
import json
import subprocess

from processor import process_session

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "..", "recordings")

os.makedirs(RECORDINGS_DIR, exist_ok=True)


# --- FRONTEND ROUTES ---
@app.route("/app")
def index_page():
    return render_template("index.html")


@app.route("/result")
def result_page():
    return render_template("result.html")


# --- VIDEO SERVE ---
@app.route("/video/<session_id>")
def get_video(session_id):
    video_path = os.path.join(RECORDINGS_DIR, session_id, "video.webm")

    if not os.path.exists(video_path):
        return "Video not found", 404

    return send_file(video_path, mimetype="video/webm")


# --- RESULTS API ---
@app.route("/results/<session_id>")
def get_results(session_id):
    results_path = os.path.join(RECORDINGS_DIR, session_id, "results.json")

    if not os.path.exists(results_path):
        return jsonify({"error": "Results not found"}), 404

    with open(results_path, "r") as f:
        return jsonify(json.load(f))


# --- UPLOAD + PROCESS ---
@app.route("/upload", methods=["POST"])
def upload_video():
    try:
        file = request.files["video"]

        session_id = str(uuid.uuid4())
        session_path = os.path.join(RECORDINGS_DIR, session_id)
        os.makedirs(session_path, exist_ok=True)

        video_path = os.path.join(session_path, "video.webm")
        audio_path = os.path.join(session_path, "audio.wav")

        file.save(video_path)

        print("[INFO] Video saved")

        # FFMPEG extraction (stable)
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            "-y",
            audio_path
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print("FFMPEG ERROR:", result.stderr.decode())
            return jsonify({"error": "Audio extraction failed"}), 500

        print("[INFO] Audio extracted")

        results = process_session(session_path)

        print("[INFO] Processing done")

        return jsonify({
            "session_id": session_id,
            "results": results
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500


# --- SESSION LIST API ---
@app.route("/api/sessions")
def list_sessions():
    """Returns a list of all past sessions with basic metadata."""
    sessions = []

    try:
        if os.path.exists(RECORDINGS_DIR):
            for session_id in os.listdir(RECORDINGS_DIR):
                session_path = os.path.join(RECORDINGS_DIR, session_id)
                results_path = os.path.join(session_path, "results.json")

                if os.path.isdir(session_path) and os.path.exists(results_path):
                    try:
                        mtime = os.path.getmtime(results_path)

                        with open(results_path, "r") as f:
                            data = json.load(f)

                        summary = data.get("summary", {})
                        sessions.append({
                            "session_id": session_id,
                            "timestamp": mtime,
                            "fluency_score": summary.get("fluency_score", None),
                            "total_events": summary.get("total_events", 0),
                            "duration": summary.get("duration", 0)
                        })
                    except Exception:
                        continue

        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        sessions = sessions[:20]

    except Exception as e:
        print(f"[ERROR] listing sessions: {e}")

    return jsonify({"sessions": sessions})


# --- CLEAN SPEECH GENERATION (NEW FEATURE) ---
@app.route("/generate-clean/<session_id>", methods=["POST"])
def generate_clean_speech(session_id):
    """
    Generates a clean/fluent version of the recorded speech.
    Pipeline: audio.wav -> Whisper STT -> gTTS TTS -> clean_speech.mp3
    """
    try:
        session_path = os.path.join(RECORDINGS_DIR, session_id)
        audio_path = os.path.join(session_path, "audio.wav")
        clean_path = os.path.join(session_path, "clean_speech.mp3")

        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404

        # If already generated, return immediately
        if os.path.exists(clean_path):
            return jsonify({
                "status": "ready",
                "text": _load_transcript(session_path),
                "url": f"/clean-audio/{session_id}"
            })

        # Step 1: Speech-to-Text using Whisper
        print(f"[INFO] Transcribing audio for session {session_id}...")
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language="en")
        transcript = result["text"].strip()

        print(f"[INFO] Transcript: {transcript[:100]}...")

        if not transcript:
            return jsonify({"error": "Could not transcribe any speech from the recording."}), 400

        # Save transcript
        transcript_path = os.path.join(session_path, "transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        # Step 2: Text-to-Speech using gTTS
        print(f"[INFO] Generating clean speech...")
        from gtts import gTTS

        tts = gTTS(text=transcript, lang="en", slow=False)
        tts.save(clean_path)

        print(f"[INFO] Clean speech saved to {clean_path}")

        return jsonify({
            "status": "ready",
            "text": transcript,
            "url": f"/clean-audio/{session_id}"
        })

    except Exception as e:
        print(f"[ERROR] Clean speech generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clean-audio/<session_id>")
def serve_clean_audio(session_id):
    """Serve the generated clean speech audio file."""
    clean_path = os.path.join(RECORDINGS_DIR, session_id, "clean_speech.mp3")

    if not os.path.exists(clean_path):
        return "Clean audio not found", 404

    return send_file(clean_path, mimetype="audio/mpeg", as_attachment=False)


@app.route("/download-clean/<session_id>")
def download_clean_audio(session_id):
    """Download the generated clean speech audio file."""
    clean_path = os.path.join(RECORDINGS_DIR, session_id, "clean_speech.mp3")

    if not os.path.exists(clean_path):
        return "Clean audio not found", 404

    return send_file(
        clean_path,
        mimetype="audio/mpeg",
        as_attachment=True,
        download_name=f"fluent_speech_{session_id[:8]}.mp3"
    )


def _load_transcript(session_path):
    """Helper to load saved transcript text."""
    transcript_path = os.path.join(session_path, "transcript.txt")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


if __name__ == "__main__":
    app.run(debug=True)