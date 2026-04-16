import asyncio
import logging
import os
from typing import Any, Dict

from .audio_extraction import extract_audio_from_video
from .stutter_detection import run_stutter_detection
from .transcription import transcribe_audio

logger = logging.getLogger(__name__)


async def run_stutter_analysis(file_path: str) -> Dict[str, Any]:
    """
    Full analysis pipeline for an uploaded video file:

    1. Extract audio from the video (16 kHz mono WAV) using ffmpeg.
    2. Run Whisper transcription and HF stutter detection concurrently in a
       thread pool so the async event loop is never blocked.
    3. Aggregate results and return a dict that matches the schema expected by
       ``dashboard.py / process_and_store()``.

    Return schema::

        {
            "fluencyScore":   float,   # 0-100, derived from model predictions
            "stutterEvents":  list,    # [{timestamp, start, end, type, confidence}, …]
            "headMovements":  list,    # kept empty – no visual model in this backend
            "transcript":     str,
            "totalWords":     int,
        }
    """
    # Step 1 — audio extraction (blocking I/O; run in thread pool)
    audio_path = await asyncio.to_thread(extract_audio_from_video, file_path)

    try:
        # Step 2 — transcription and stutter detection run concurrently
        transcript, detection = await asyncio.gather(
            asyncio.to_thread(transcribe_audio, audio_path),
            asyncio.to_thread(run_stutter_detection, audio_path),
        )
    finally:
        # Always clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    stutter_events: list = detection["events"]
    fluency_score: float = detection["fluency_score"]
    total_words: int = len(transcript.split()) if transcript else 0

    return {
        "fluencyScore": fluency_score,
        "stutterEvents": stutter_events,
        "headMovements": [],
        "transcript": transcript,
        "totalWords": total_words,
    }
