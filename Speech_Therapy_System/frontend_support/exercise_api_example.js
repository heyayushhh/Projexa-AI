/**
 * Exercise API – Frontend Integration Example
 *
 * This file demonstrates how a React / vanilla JS frontend can call the
 * Speech Therapy Exercise Recommendation API to fetch exercises based on
 * the stuttering type and severity returned by the detection model.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_BASE_URL = "http://localhost:5000";

// ---------------------------------------------------------------------------
// 1.  Vanilla JavaScript (fetch)
// ---------------------------------------------------------------------------

/**
 * Fetch recommended exercises from the backend.
 *
 * @param {string} type     - Stuttering type, e.g. "WordRep" or "1"
 * @param {string} severity - Severity level, e.g. "mild", "moderate", "severe"
 * @returns {Promise<Object>} The exercise data returned by the API.
 */
async function getExercises(type, severity) {
  try {
    const response = await fetch(`${API_BASE_URL}/get_exercises`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ type, severity }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || "Failed to fetch exercises.");
    }

    const data = await response.json();
    console.log("Exercises received:", data);
    return data;
  } catch (error) {
    console.error("Error fetching exercises:", error.message);
    throw error;
  }
}

// Example usage:
// getExercises("WordRep", "moderate").then(data => console.log(data));

// ---------------------------------------------------------------------------
// 2.  React Component Example (functional component with hooks)
// ---------------------------------------------------------------------------

/*
import React, { useState } from "react";

function ExerciseRecommendation() {
  const [exercises, setExercises] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // This would normally receive values from the stuttering detection model.
  const stutterType = "WordRep";
  const severity = "moderate";

  const fetchExercises = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/get_exercises", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          type: stutterType,
          severity: severity,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Failed to fetch exercises.");
      }

      const data = await response.json();
      setExercises(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "24px", fontFamily: "sans-serif" }}>
      <h2>Speech Therapy Exercises</h2>

      <button onClick={fetchExercises} disabled={loading}>
        {loading ? "Loading…" : "Get Exercises"}
      </button>

      {error && <p style={{ color: "red" }}>Error: {error}</p>}

      {exercises && (
        <div style={{ marginTop: "16px" }}>
          <h3>
            {exercises.stutter_type} — {exercises.level}
          </h3>

          {exercises.exercises.map((ex, index) => (
            <div
              key={index}
              style={{
                border: "1px solid #ddd",
                borderRadius: "8px",
                padding: "16px",
                marginBottom: "12px",
              }}
            >
              <h4>{ex.name}</h4>
              <p>
                <strong>Instruction:</strong> {ex.instruction}
              </p>
              <p>
                <strong>Duration:</strong> {ex.duration}
              </p>
              <div>
                <strong>Practice Material:</strong>
                <ul>
                  {ex.practice_material.map((item, i) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ExerciseRecommendation;
*/
