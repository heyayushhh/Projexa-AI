"""
Stutter detection service using the HareemFatima/distilhubert-finetuned-stutterdetection
model from Hugging Face.

The model and feature extractor are loaded once as module-level singletons so
they are not reloaded on every request.
"""

import logging
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import torch
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)

_MODEL_NAME = "HareemFatima/distilhubert-finetuned-stutterdetection"

# Integer label → string label mapping used by the HF model
_LABEL_MAP: Dict[int, str] = {
    0: "fluent",
    1: "prolongation",
    2: "block",
    3: "sound_rep",
    4: "word_rep",
    5: "difficult",
    6: "interjection",
}

# How much each stutter type penalises the fluency score (higher = more severe)
_SEVERITY_WEIGHTS: Dict[str, float] = {
    "fluent": 0.0,
    "prolongation": 0.6,
    "block": 0.8,
    "sound_rep": 0.5,
    "word_rep": 0.5,
    "difficult": 0.9,
    "interjection": 0.3,
}

_CONFIDENCE_THRESHOLD: float = 0.55
_CHUNK_DURATION: int = 2              # seconds per inference window
_MIN_TAIL_DURATION: float = 0.5      # discard tail chunks shorter than this (seconds)
_MAX_EVENTS_PER_MINUTE: float = 20.0 # baseline for density normalisation
_SEVERITY_NORMALIZATION: float = 10.0 # denominator when scaling severity score

# Weights for the three fluency sub-scores (must sum to 1.0)
_WEIGHT_DENSITY: float = 0.35
_WEIGHT_SEVERITY: float = 0.35
_WEIGHT_TIME: float = 0.30

# Module-level singletons — populated lazily on first call
_processor: Any = None
_model: Any = None
_device: Any = None


def _ensure_model_loaded() -> None:
    """Load the HF processor and model if they have not been loaded yet."""
    global _processor, _model, _device
    if _processor is not None:
        return

    logger.info("Loading stutter detection model (%s)…", _MODEL_NAME)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _processor = Wav2Vec2FeatureExtractor.from_pretrained(_MODEL_NAME)
    _model = AutoModelForAudioClassification.from_pretrained(_MODEL_NAME)
    _model.to(_device)
    _model.eval()
    logger.info("Stutter detection model loaded on %s", _device)


def _load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to 16 kHz mono."""
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio, sr


def _chunk_audio(
    audio: np.ndarray, sr: int
) -> List[Tuple[float, np.ndarray]]:
    """
    Divide the audio signal into fixed-length windows of *_CHUNK_DURATION* seconds.

    - Full-length chunks are passed through unchanged.
    - Tail chunks that are shorter than the window but at least 0.5 s long are
      zero-padded to the full window length so they are not silently dropped.
    - Tail chunks shorter than 0.5 s are discarded.

    Returns a list of (start_time_seconds, chunk_array) tuples.
    """
    chunk_size = int(sr * _CHUNK_DURATION)
    chunks: List[Tuple[float, np.ndarray]] = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        start_time = i / sr

        if len(chunk) < chunk_size:
            # Only keep if at least _MIN_TAIL_DURATION of real audio
            if len(chunk) >= int(sr * _MIN_TAIL_DURATION):
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunks.append((start_time, padded))
        else:
            chunks.append((start_time, chunk))

    return chunks


def _calculate_fluency_score(events: List[dict], total_duration: float) -> float:
    """
    Produce a fluency score in [0, 100] where 100 means perfectly fluent.

    Three equally-weighted sub-scores are combined:
      1. Density score   — penalises a high number of events per minute.
      2. Severity score  — penalises severe stutter types weighted by confidence.
      3. Time score      — penalises a high proportion of stuttered time.
    """
    if total_duration <= 0 or not events:
        return 100.0

    # 1. Density
    events_per_minute = (len(events) / total_duration) * 60
    density_score = max(0.0, 1.0 - (events_per_minute / _MAX_EVENTS_PER_MINUTE))

    # 2. Severity
    total_severity = sum(
        _SEVERITY_WEIGHTS.get(e["type"], 0.5) * e.get("confidence", 0.7)
        for e in events
    )
    severity_score = max(0.0, 1.0 - (total_severity / _SEVERITY_NORMALIZATION))

    # 3. Stutter time ratio
    stutter_time = sum(e["end"] - e["start"] for e in events)
    time_score = max(0.0, 1.0 - (stutter_time / total_duration))

    raw = (density_score * _WEIGHT_DENSITY) + (severity_score * _WEIGHT_SEVERITY) + (time_score * _WEIGHT_TIME)
    return round(raw * 100, 1)


def run_stutter_detection(audio_path: str) -> Dict[str, Any]:
    """
    Run end-to-end stutter detection on a 16 kHz mono WAV file.

    Returns::

        {
            "events": [
                {
                    "timestamp": <float>,   # chunk start, for frontend compat
                    "start":     <float>,
                    "end":       <float>,
                    "type":      <str>,
                    "confidence":<float>,
                },
                …
            ],
            "fluency_score": <float>,  # 0-100
        }
    """
    _ensure_model_loaded()

    audio, sr = _load_audio(audio_path)
    total_duration = len(audio) / sr
    chunks = _chunk_audio(audio, sr)

    events: List[dict] = []
    for start_time, chunk in chunks:
        inputs = _processor(
            chunk,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = _model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence: float = probabilities.max().item()
            pred: int = torch.argmax(logits, dim=-1).item()

        label = _LABEL_MAP.get(pred, "unknown")

        if label != "fluent" and confidence >= _CONFIDENCE_THRESHOLD:
            events.append(
                {
                    # "timestamp" keeps backward compatibility with the existing
                    # frontend / DB schema that uses stutterEvents[].timestamp
                    "timestamp": round(start_time, 2),
                    "start": round(start_time, 2),
                    "end": round(start_time + _CHUNK_DURATION, 2),
                    "type": label,
                    "confidence": round(confidence, 3),
                }
            )

    fluency_score = _calculate_fluency_score(events, total_duration)
    return {"events": events, "fluency_score": fluency_score}
