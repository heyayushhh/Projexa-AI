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


# ✅ FRONTEND ROUTES
@app.route("/app")
def index_page():
    return render_template("index.html")


@app.route("/result")
def result_page():
    return render_template("result.html")


# ✅ VIDEO SERVE
@app.route("/video/<session_id>")
def get_video(session_id):
    video_path = os.path.join(RECORDINGS_DIR, session_id, "video.webm")

    if not os.path.exists(video_path):
        return "Video not found", 404

    return send_file(video_path)


# ✅ RESULTS API
@app.route("/results/<session_id>")
def get_results(session_id):
    results_path = os.path.join(RECORDINGS_DIR, session_id, "results.json")

    if not os.path.exists(results_path):
        return jsonify({"error": "Results not found"}), 404

    with open(results_path, "r") as f:
        return jsonify(json.load(f))


# ✅ UPLOAD + PROCESS
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

        # 🔥 FFMPEG extraction (stable)
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


if __name__ == "__main__":
    app.run(debug=True)