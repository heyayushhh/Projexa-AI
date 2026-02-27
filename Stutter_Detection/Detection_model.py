import torch
import numpy as np
import sounddevice as sd
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import scipy.signal

# Load model + feature extractor
model_name = "HareemFatima/distilhubert-finetuned-stutterdetection"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

model.eval()

# Audio settings
SAMPLE_RATE = 16000  # Required for HuBERT
DURATION = 3         # seconds per chunk

def record_audio(duration, sample_rate):
    print("Listening...")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='float32')
    sd.wait()
    return audio.squeeze()

def detect_stutter(audio):
    inputs = feature_extractor(audio,
                               sampling_rate=SAMPLE_RATE,
                               return_tensors="pt",
                               padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    predicted_class = torch.argmax(probs).item()
    confidence = probs[0][predicted_class].item()

    label = model.config.id2label[predicted_class]

    return label, confidence


print("Live Stutter Detection Started (Ctrl+C to stop)\n")

while True:
    audio_chunk = record_audio(DURATION, SAMPLE_RATE)
    label, confidence = detect_stutter(audio_chunk)
    print(f"Prediction: {label} | Confidence: {confidence:.2f}\n")




# pip install torch torchaudio transformers sounddevice numpy scipy
