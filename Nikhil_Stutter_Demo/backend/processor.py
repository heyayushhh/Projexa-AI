import os
import json
from audio_model import AudioStutterDetector
from visual_model import VisualStutterDetector

audio_detector = AudioStutterDetector()
visual_detector = VisualStutterDetector()


def process_session(session_path):
    audio_path = os.path.join(session_path, "audio.wav")
    video_path = os.path.join(session_path, "video.webm")
    output_path = os.path.join(session_path, "results.json")

    print("Processing audio...")
    audio_results = audio_detector.predict(audio_path)

    print("Processing video...")
    visual_results = visual_detector.detect(video_path)

    # 🔥 MERGE RESULTS
    combined = audio_results + visual_results

    combined.sort(key=lambda x: x["start"])

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=4)

    return combined