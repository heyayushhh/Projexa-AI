import torch
import numpy as np
import sounddevice as sd
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from collections import deque

# Load model
model_name = "HareemFatima/distilhubert-finetuned-stutterdetection"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Audio settings
SAMPLE_RATE = 16000
WINDOW_SIZE = 3
STEP_SIZE = 1

buffer = np.zeros(SAMPLE_RATE * WINDOW_SIZE)

prediction_history = deque(maxlen=5)

def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

def detect(audio):

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    inputs = {k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pred = torch.argmax(probs).item()

    label = model.config.id2label[pred]

    return label

def majority_vote():

    if len(prediction_history) == 0:
        return None

    return max(set(prediction_history), key=prediction_history.count)

print("Real-time stutter detection started")

while True:

    audio = sd.rec(int(STEP_SIZE * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)

    sd.wait()

    audio = audio.squeeze()

    audio = normalize(audio)

    buffer = np.roll(buffer, -len(audio))
    buffer[-len(audio):] = audio

    label = detect(buffer)

    prediction_history.append(label)

    final_prediction = majority_vote()

    print("Prediction:", final_prediction)