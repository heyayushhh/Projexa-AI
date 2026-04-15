import asyncio
import logging
from typing import Dict, List, Any, Tuple

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

logger = logging.getLogger(__name__)

MODEL_NAME = "HareemFatima/distilhubert-finetuned-stutterdetection"

LABEL_MAP = {
    0: "fluent",
    1: "prolongation",
    2: "block",
    3: "sound_rep",
    4: "word_rep",
    5: "difficult",
    6: "interjection",
}

SEVERITY_WEIGHTS = {
    "prolongation": 0.6,
    "block": 0.8,
    "sound_rep": 0.5,
    "word_rep": 0.5,
    "difficult": 0.9,
    "interjection": 0.3,
}

_model_lock = asyncio.Lock()
_feature_extractor = None
_model = None
_device = None


async def _load_model_if_needed() -> None:
    global _feature_extractor, _model, _device
    if _model is not None and _feature_extractor is not None:
        return
    async with _model_lock:
        if _model is not None and _feature_extractor is not None:
            return

        def _load_sync():
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
            model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            return feature_extractor, model, device

        _feature_extractor, _model, _device = await asyncio.to_thread(_load_sync)
        logger.info("Loaded stutter model '%s' on %s", MODEL_NAME, _device)


def _chunk_audio(audio: np.ndarray, sr: int, chunk_duration: int = 2) -> List[Tuple[float, np.ndarray]]:
    chunk_size = int(sr * chunk_duration)
    chunks: List[Tuple[float, np.ndarray]] = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            if len(chunk) > int(sr * 0.5):
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunks.append((i / sr, padded))
        else:
            chunks.append((i / sr, chunk))
    return chunks


def _calculate_fluency_score(events: List[Dict[str, Any]], total_duration: float) -> int:
    if total_duration <= 0 or not events:
        return 100

    events_per_minute = (len(events) / total_duration) * 60
    density_score = max(0.0, 1.0 - (events_per_minute / 20.0))

    total_severity = 0.0
    for event in events:
        severity = SEVERITY_WEIGHTS.get(event["type"], 0.5)
        confidence = event.get("confidence", 0.7)
        total_severity += severity * confidence
    severity_score = max(0.0, 1.0 - (total_severity / 10.0))

    stutter_time = sum(event["end"] - event["start"] for event in events)
    stutter_ratio = stutter_time / total_duration
    time_score = max(0.0, 1.0 - stutter_ratio)

    combined = (density_score * 0.35) + (severity_score * 0.35) + (time_score * 0.30)
    return round(combined * 100)


def _predict_stutter_events(file_path: str, confidence_threshold: float = 0.55) -> Tuple[List[Dict[str, Any]], float]:
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    chunks = _chunk_audio(audio, sr)
    total_duration = len(audio) / sr if sr else 0.0

    events: List[Dict[str, Any]] = []
    for start_time, chunk in chunks:
        inputs = _feature_extractor(
            chunk,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = _model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence = probabilities.max().item()
            pred_idx = torch.argmax(logits, dim=-1).item()
        label = LABEL_MAP.get(pred_idx, "fluent")
        if label != "fluent" and confidence >= confidence_threshold:
            events.append(
                {
                    "start": round(start_time, 2),
                    "end": round(start_time + 2, 2),
                    "type": label,
                    "confidence": round(confidence, 3),
                }
            )
    return events, total_duration


async def run_stutter_analysis(file_path: str) -> Dict[str, Any]:
    await _load_model_if_needed()

    def _run_sync():
        events, total_duration = _predict_stutter_events(file_path)
        fluency_score = _calculate_fluency_score(events, total_duration)
        timeline_events = [
            {"timestamp": event["start"], "type": event["type"], "confidence": event["confidence"]}
            for event in events
        ]
        transcript = "Transcript generation is not enabled in the backend yet."
        return {
            "fluencyScore": fluency_score,
            "stutterEvents": timeline_events,
            "headMovements": [],
            "transcript": transcript,
            "totalWords": len(transcript.split()),
        }

    return await asyncio.to_thread(_run_sync)
