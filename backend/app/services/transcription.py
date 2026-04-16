"""
Speech transcription service using OpenAI Whisper.

The Whisper model is loaded once as a module-level singleton so it is not
reloaded on every request.  The "base" model is chosen as a sensible default
that runs acceptably on CPU; swap it for "small" or "medium" if higher
accuracy is needed and the hardware supports it.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_WHISPER_MODEL_SIZE = "base"

# Module-level singleton — populated lazily on first call
_whisper_model: Any = None


def _ensure_model_loaded() -> Any:
    """Load Whisper if it has not been loaded yet and return the model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper  # imported here so the module can be imported without whisper installed

        logger.info("Loading Whisper model (%s)…", _WHISPER_MODEL_SIZE)
        _whisper_model = whisper.load_model(_WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded.")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using Whisper.

    Returns the transcribed text as a string, or an empty string if
    transcription fails (so that stutter detection can still proceed).
    """
    try:
        model = _ensure_model_loaded()
        result = model.transcribe(audio_path)
        text: str = result.get("text", "").strip()
        logger.info("Transcription complete (%d chars).", len(text))
        return text
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return ""
