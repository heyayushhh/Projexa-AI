import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForAudioClassification

model_name = "Huma10/Whisper_Stuttered_Speech"

print("Loading model...")

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    use_safetensors=False
)

audio_path = "audio_stutter.wav"

audio, sr = librosa.load(audio_path, sr=16000)

# chunk length in seconds
chunk_length = 3
samples_per_chunk = chunk_length * sr

num_chunks = int(np.ceil(len(audio) / samples_per_chunk))

print("\nAnalyzing audio...\n")

predictions = []

# -------- CHUNK PREDICTIONS --------

for i in range(num_chunks):

    start_sample = i * samples_per_chunk
    end_sample = start_sample + samples_per_chunk

    chunk = audio[start_sample:end_sample]

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

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()

    label = model.config.id2label[predicted_class_id]

    start_time = i * chunk_length
    end_time = start_time + chunk_length

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

# last segment
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
