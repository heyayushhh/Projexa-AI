"""
Speech Therapy Exercise Recommendation API
Flask backend that serves exercises based on stuttering type and severity.
"""

import json
import os

from flask import Flask, jsonify, request
from severity_mapper import severity_to_level

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the exercise database once at startup
# ---------------------------------------------------------------------------
EXERCISE_DB_PATH = os.path.join(os.path.dirname(__file__), "exercise_db.json")

with open(EXERCISE_DB_PATH, "r", encoding="utf-8") as f:
    EXERCISE_DB = json.load(f)

# Mapping from numeric type codes (sent by the detection model) to keys
# used inside exercise_db.json
TYPE_CODE_MAP = {
    "0": "NoStutteredWords",
    "1": "WordRep",
    "2": "SoundRep",
    "3": "Prolongation",
    "4": "Interjection",
    "5": "Block",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/get_exercises", methods=["POST"])
def get_exercises():
    """
    Receive a JSON payload with the detected stuttering type and severity,
    then return the matching exercises from the database.

    Expected JSON body:
        {
            "type": "WordRep"  or  "1",
            "severity": "moderate"
        }

    Returns:
        200 – JSON list of exercises for the resolved type & level.
        400 – If required fields are missing or values are invalid.
    """
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    stutter_type = data.get("type")
    severity = data.get("severity")

    if not stutter_type or not severity:
        return jsonify({"error": "'type' and 'severity' fields are required."}), 400

    # Accept both numeric codes ("1") and string names ("WordRep")
    stutter_type_key = TYPE_CODE_MAP.get(str(stutter_type), stutter_type)

    # Validate the stuttering type
    if stutter_type_key not in EXERCISE_DB:
        valid_types = list(EXERCISE_DB.keys()) + list(TYPE_CODE_MAP.keys())
        return jsonify({
            "error": f"Unknown stuttering type '{stutter_type}'.",
            "valid_types": valid_types,
        }), 400

    # Convert severity to exercise level
    try:
        level = severity_to_level(severity)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Fetch exercises
    exercises = EXERCISE_DB[stutter_type_key].get(level, [])

    return jsonify({
        "stutter_type": stutter_type_key,
        "severity": severity,
        "level": level,
        "exercises": exercises,
    }), 200


@app.route("/types", methods=["GET"])
def list_types():
    """Return all available stuttering types and their numeric codes."""
    return jsonify({
        "types": list(EXERCISE_DB.keys()),
        "code_map": TYPE_CODE_MAP,
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
