import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "HareemFatima/distilhubert-finetuned-stutterdetection"

class AudioStutterDetector:
    def __init__(self):
        print("Loading audio model...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

        self.label_map = {
            0: "fluent",
            1: "prolongation",
            2: "block",
            3: "sound_rep",
            4: "word_rep",
            5: "difficult",
            6: "interjection"
        }

    def load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        return audio, sr

    def chunk_audio(self, audio, sr, chunk_duration=2):
        chunk_size = int(sr * chunk_duration)
        chunks = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) == chunk_size:
                chunks.append((i / sr, chunk))  # timestamp

        return chunks

    def predict(self, audio_path):
        audio, sr = self.load_audio(audio_path)
        chunks = self.chunk_audio(audio, sr)

        results = []

        for start_time, chunk in chunks:
            inputs = self.processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits = self.model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()

            label = self.label_map[pred]

            if label != "fluent":
                results.append({
                    "start": round(start_time, 2),
                    "end": round(start_time + 2, 2),
                    "type": label
                })

        return results