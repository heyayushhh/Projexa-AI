import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "HareemFatima/distilhubert-finetuned-stutterdetection"


class AudioStutterDetector:
    def __init__(self):
        print("Loading audio model...")

        # Detect best available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Audio model using device: {self.device}")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
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

        # Friendly display names for each stutter type
        self.friendly_names = {
            "fluent": "Fluent Speech",
            "prolongation": "Prolongation",
            "block": "Block",
            "sound_rep": "Sound Repetition",
            "word_rep": "Word Repetition",
            "difficult": "Difficult Speech",
            "interjection": "Interjection",
            "head_jerk": "Head Movement"
        }

        # Severity weights for fluency score calculation (higher = more severe)
        self.severity_weights = {
            "fluent": 0,
            "prolongation": 0.6,
            "block": 0.8,
            "sound_rep": 0.5,
            "word_rep": 0.5,
            "difficult": 0.9,
            "interjection": 0.3
        }

        # Minimum confidence to report an event
        self.confidence_threshold = 0.55

    def load_audio(self, file_path):
        """Load audio file and resample to 16kHz."""
        audio, sr = librosa.load(file_path, sr=16000)
        return audio, sr

    def chunk_audio(self, audio, sr, chunk_duration=2):
        """
        Split audio into fixed-length chunks for classification.
        Now handles tail chunks by zero-padding instead of discarding.
        """
        chunk_size = int(sr * chunk_duration)
        chunks = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]

            # Pad short tail chunks with zeros instead of discarding
            if len(chunk) < chunk_size:
                if len(chunk) > sr * 0.5:  # Only if chunk is at least 0.5s
                    padded = np.zeros(chunk_size, dtype=np.float32)
                    padded[:len(chunk)] = chunk
                    chunks.append((i / sr, padded))
            else:
                chunks.append((i / sr, chunk))

        return chunks

    def predict(self, audio_path):
        """
        Run stutter detection on an audio file.
        Returns a list of detected disfluency events with confidence scores.
        Preserves original return format: list of dicts with start, end, type.
        Extends with: confidence, source fields.
        """
        audio, sr = self.load_audio(audio_path)
        chunks = self.chunk_audio(audio, sr)

        # Calculate total audio duration for metadata
        total_duration = len(audio) / sr

        results = []

        for start_time, chunk in chunks:
            inputs = self.processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

                # Get softmax probabilities for confidence scoring
                probabilities = torch.softmax(logits, dim=-1)
                confidence = probabilities.max().item()
                pred = torch.argmax(logits, dim=-1).item()

            label = self.label_map[pred]

            # Only report non-fluent events above confidence threshold
            if label != "fluent" and confidence >= self.confidence_threshold:
                results.append({
                    "start": round(start_time, 2),
                    "end": round(start_time + 2, 2),
                    "type": label,
                    "confidence": round(confidence, 3),
                    "source": "audio"
                })

        return results