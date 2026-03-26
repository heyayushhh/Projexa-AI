import torch
import librosa
import numpy as np
import sounddevice as sd
import time
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForAudioClassification

model_name = "Huma10/Whisper_Stuttered_Speech"

print("Loading model...")

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    use_safetensors=False
)

# ---------------- MIC RECORDING ----------------

sr = 16000
device_id = 1   # Airdopes 280 ANC mic

input("Press ENTER to start speaking...")

print("Recording... Press ENTER to stop")

recording = []

def callback(indata, frames, time_info, status):
    recording.append(indata.copy())

stream = sd.InputStream(
    samplerate=sr,
    channels=1,
    device=device_id,
    callback=callback
)

with stream:
    input()

print("Recording stopped")

audio = np.concatenate(recording, axis=0).flatten()

print("Recording duration:", round(len(audio)/sr,2), "seconds")

# ---------------- CHUNK SETTINGS ----------------

chunk_length = 1        # improved detection
samples_per_chunk = chunk_length * sr

confidence_threshold = 0.6
silence_threshold = 0.01

num_chunks = int(np.ceil(len(audio) / samples_per_chunk))

print("\nAnalyzing audio...\n")

predictions = []

# -------- CHUNK PREDICTIONS --------

for i in range(num_chunks):

    start_sample = i * samples_per_chunk
    end_sample = start_sample + samples_per_chunk

    chunk = audio[start_sample:end_sample]

    start_time = i * chunk_length
    end_time = start_time + chunk_length

    # Silence filter
    if np.mean(np.abs(chunk)) < silence_threshold:

        label = "NoStutteredWords"

    else:

        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=30 * 16000
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)
        confidence, predicted_class_id = torch.max(probs, dim=-1)

        confidence = confidence.item()
        predicted_class_id = predicted_class_id.item()

        label = model.config.id2label[predicted_class_id]

        # Confidence filter
        if confidence < confidence_threshold:
            label = "NoStutteredWords"

    predictions.append({
        "label": label,
        "start": start_time,
        "end": end_time
    })

    print(f"{start_time}s - {end_time}s : {label}")

# -------- SEGMENT MERGING --------

segments = []

current_label = predictions[0]["label"]
segment_start = predictions[0]["start"]

for i in range(1, len(predictions)):

    label = predictions[i]["label"]

    if label != current_label:

        segments.append({
            "label": current_label,
            "start": segment_start,
            "end": predictions[i-1]["end"]
        })

        current_label = label
        segment_start = predictions[i]["start"]

segments.append({
    "label": current_label,
    "start": segment_start,
    "end": predictions[-1]["end"]
})

# -------- STUTTER SCORING --------

severity_weights = {
    "NoStutteredWords": 0,
    "Interjection": 1,
    "WordRep": 2,
    "SoundRep": 3,
    "Prolongation": 3,
    "Block": 5
}

score = 0
max_score = 0

print("\nDetected Stutter Segments:\n")

for seg in segments:

    duration = seg["end"] - seg["start"]
    weight = severity_weights.get(seg["label"], 0)

    segment_score = weight * duration

    score += segment_score
    max_score += 5 * duration

    if seg["label"] != "NoStutteredWords":
        print(f'{seg["label"]} : {seg["start"]}s - {seg["end"]}s')

# -------- FINAL INTENSITY SCORE --------

intensity = score / max_score if max_score > 0 else 0

print("\nStuttering Intensity Score:", round(intensity, 3))

# -------- SEVERITY LEVEL --------

if intensity < 0.15:
    severity = "Fluent"
elif intensity < 0.35:
    severity = "Mild"
elif intensity < 0.6:
    severity = "Moderate"
else:
    severity = "Severe"

print("Severity Level:", severity)
