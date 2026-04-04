import os
import json
import threading
import librosa
from audio_model import AudioStutterDetector
from visual_model import VisualStutterDetector

audio_detector = AudioStutterDetector()
visual_detector = VisualStutterDetector()

# Severity weights for fluency score calculation
SEVERITY_WEIGHTS = {
    "prolongation": 0.6,
    "block": 0.8,
    "sound_rep": 0.5,
    "word_rep": 0.5,
    "difficult": 0.9,
    "interjection": 0.3,
    "head_jerk": 0.2
}

# Friendly display names
FRIENDLY_NAMES = {
    "prolongation": "Prolongation",
    "block": "Block",
    "sound_rep": "Sound Repetition",
    "word_rep": "Word Repetition",
    "difficult": "Difficult Speech",
    "interjection": "Interjection",
    "head_jerk": "Head Movement"
}

# Stutter type colors for frontend
STUTTER_COLORS = {
    "prolongation": "#E8A838",
    "block": "#E05D5D",
    "sound_rep": "#7C6BB5",
    "word_rep": "#5A6ACF",
    "difficult": "#C44040",
    "interjection": "#2BA882",
    "head_jerk": "#2AA5BD"
}

# Suggestions per stutter type
SUGGESTIONS = {
    "block": [
        "Try gentle onset \u2014 start speaking with a soft, easy voice.",
        "Practice light articulatory contacts to reduce blocking.",
        "Pause and breathe before starting difficult words."
    ],
    "prolongation": [
        "Practice easy stretching \u2014 gently prolonging initial sounds.",
        "Try reducing time pressure when speaking.",
        "Focus on smooth, continuous airflow."
    ],
    "sound_rep": [
        "Slow down slightly and let the word come naturally.",
        "Practice easy repetitions \u2014 gentle bouncing on sounds.",
        "Try continuous phonation through the repeated sound."
    ],
    "word_rep": [
        "Pause briefly, then try the sentence again slowly.",
        "Use phrasing \u2014 break speech into shorter, comfortable groups.",
        "Practice mindful speaking at a comfortable pace."
    ],
    "interjection": [
        "It\u2019s okay to pause instead of filling silence.",
        "Practice comfortable silence between phrases.",
        "Try slowing down \u2014 fillers often happen when we rush."
    ],
    "difficult": [
        "Take your time \u2014 there\u2019s no rush.",
        "Practice diaphragmatic breathing before speaking.",
        "Try using shorter sentences when feeling tense."
    ],
    "head_jerk": [
        "Practice relaxed posture while speaking.",
        "Try progressive muscle relaxation before speaking tasks.",
        "Focus on releasing tension in your neck and shoulders."
    ]
}


def calculate_fluency_score(events, total_duration):
    """
    Calculate a fluency score from 0-100 (100 = fully fluent).
    Factors: event density, severity weighting, stutter time ratio.
    """
    if total_duration <= 0:
        return 100

    if not events:
        return 100

    # Factor 1: Event density (events per minute)
    events_per_minute = (len(events) / total_duration) * 60
    density_score = max(0, 1.0 - (events_per_minute / 20))

    # Factor 2: Severity-weighted score
    total_severity = 0
    for e in events:
        weight = SEVERITY_WEIGHTS.get(e["type"], 0.5)
        confidence = e.get("confidence", 0.7)
        total_severity += weight * confidence

    severity_score = max(0, 1.0 - (total_severity / 10))

    # Factor 3: Stutter time ratio
    stutter_time = sum(e["end"] - e["start"] for e in events)
    stutter_ratio = stutter_time / total_duration
    time_score = max(0, 1.0 - stutter_ratio)

    # Weighted combination
    raw_score = (density_score * 0.35) + (severity_score * 0.35) + (time_score * 0.30)

    return round(raw_score * 100)


def apply_cooccurrence_fusion(audio_results, visual_results):
    """
    Boost confidence of audio events co-occurring with visual events.
    Dampen isolated visual events.
    """
    for audio_event in audio_results:
        for visual_event in visual_results:
            overlap_start = max(audio_event["start"], visual_event["start"])
            overlap_end = min(audio_event["end"], visual_event["end"])

            if overlap_start < overlap_end:
                current = audio_event.get("confidence", 0.7)
                audio_event["confidence"] = round(min(1.0, current * 1.15), 3)
                visual_event["_corroborated"] = True

    for visual_event in visual_results:
        if not visual_event.get("_corroborated", False):
            current = visual_event.get("confidence", 0.5)
            visual_event["confidence"] = round(current * 0.7, 3)
        visual_event.pop("_corroborated", None)

    return audio_results, visual_results


def generate_enhanced_insights(events, type_count, total_duration, fluency_score):
    """Generate actionable insights based on detection results."""
    insights = []
    visual_count = type_count.get("head_jerk", 0)

    # Original insights (preserved)
    if "block" in type_count:
        insights.append("You tend to pause or block during speech.")

    if "sound_rep" in type_count or "word_rep" in type_count:
        insights.append("Repetition patterns detected in your speech.")

    if visual_count > 0:
        insights.append("Facial tension or head movement detected during stuttering.")

    total = len(events)
    if total > 5:
        insights.append("High frequency of stuttering observed.")

    # Positive reinforcement (no emojis)
    if fluency_score >= 80:
        insights.insert(0, "Great job! Your speech was mostly fluent today.")
    elif fluency_score >= 60:
        insights.insert(0, "You're making progress. Keep practicing.")
    elif fluency_score >= 40:
        insights.insert(0, "Every practice session helps. You're doing great by showing up.")
    else:
        insights.insert(0, "Remember \u2014 stuttering doesn't define you. This tool is here to help you grow.")

    # Pattern-specific
    if "interjection" in type_count and type_count["interjection"] >= 3:
        insights.append("Frequent filler words detected \u2014 this is very common and can be reduced with practice.")

    if "prolongation" in type_count and "block" in type_count:
        insights.append("Both prolongations and blocks detected \u2014 focused breathing exercises may help.")

    # Duration insight
    if total_duration > 0 and total > 0:
        stutter_time = sum(e["end"] - e["start"] for e in events)
        stutter_pct = (stutter_time / total_duration) * 100
        if stutter_pct < 5:
            insights.append(f"Only {stutter_pct:.1f}% of your speech showed disfluency \u2014 that's excellent.")
        elif stutter_pct < 15:
            insights.append(f"About {stutter_pct:.1f}% of your speech showed disfluency \u2014 you're in a good range.")

    return insights


def generate_suggestions(type_count):
    """Generate specific, actionable suggestions based on detected stutter types."""
    suggestions = []
    seen = set()

    sorted_types = sorted(type_count.items(), key=lambda x: x[1], reverse=True)

    for stutter_type, count in sorted_types:
        if stutter_type in SUGGESTIONS:
            for s in SUGGESTIONS[stutter_type]:
                if s not in seen:
                    suggestions.append({
                        "text": s,
                        "for_type": stutter_type,
                        "friendly_name": FRIENDLY_NAMES.get(stutter_type, stutter_type)
                    })
                    seen.add(s)

    return suggestions[:5]


def process_session(session_path):
    """
    Main processing pipeline.
    Runs audio and visual detection in parallel, then merges results.
    """
    audio_path = os.path.join(session_path, "audio.wav")
    video_path = os.path.join(session_path, "video.webm")
    output_path = os.path.join(session_path, "results.json")

    # Parallel processing
    audio_results = []
    visual_results = []
    audio_error = None
    visual_error = None

    def run_audio():
        nonlocal audio_results, audio_error
        try:
            print("Processing audio...")
            audio_results = audio_detector.predict(audio_path)
            print(f"[INFO] Audio: {len(audio_results)} events detected")
        except Exception as e:
            audio_error = str(e)
            print(f"[ERROR] Audio processing: {e}")

    def run_visual():
        nonlocal visual_results, visual_error
        try:
            print("Processing video...")
            visual_results = visual_detector.detect(video_path)
            print(f"[INFO] Visual: {len(visual_results)} events detected")
        except Exception as e:
            visual_error = str(e)
            print(f"[ERROR] Visual processing: {e}")

    audio_thread = threading.Thread(target=run_audio)
    visual_thread = threading.Thread(target=run_visual)

    audio_thread.start()
    visual_thread.start()

    audio_thread.join()
    visual_thread.join()

    # Co-occurrence fusion
    audio_results, visual_results = apply_cooccurrence_fusion(audio_results, visual_results)

    # Merge
    combined = audio_results + visual_results
    combined.sort(key=lambda x: x["start"])

    # Calculate duration (fixed deprecation warning)
    try:
        audio_duration = librosa.get_duration(path=audio_path)
    except Exception:
        audio_duration = combined[-1]["end"] if combined else 0

    # Summary
    total = len(combined)

    type_count = {}
    visual_count = 0

    for e in combined:
        t = e["type"]
        type_count[t] = type_count.get(t, 0) + 1
        if t == "head_jerk":
            visual_count += 1

    most_common = max(type_count, key=type_count.get) if type_count else "None"

    fluency_score = calculate_fluency_score(combined, audio_duration)
    insights = generate_enhanced_insights(combined, type_count, audio_duration, fluency_score)
    suggestions = generate_suggestions(type_count)

    summary = {
        "total_events": total,
        "most_common": most_common,
        "visual_events": visual_count,
        "fluency_score": fluency_score,
        "duration": round(audio_duration, 2),
        "type_breakdown": type_count,
        "audio_events": total - visual_count
    }

    final_output = {
        "events": combined,
        "summary": summary,
        "insights": insights,
        "suggestions": suggestions,
        "stutter_colors": STUTTER_COLORS,
        "friendly_names": FRIENDLY_NAMES
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

    return final_output