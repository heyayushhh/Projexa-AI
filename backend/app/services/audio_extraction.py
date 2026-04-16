import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT_SECONDS = 180  # generous upper bound for long video files


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from a video file and write it as a 16 kHz mono WAV file.

    Uses ffmpeg via subprocess so no Python binding is needed beyond having the
    ffmpeg binary available on PATH.

    Returns the path to the temporary WAV file.  The caller is responsible for
    deleting the file when done.

    Raises RuntimeError if ffmpeg is not found or returns a non-zero exit code.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",           # overwrite output without asking
        "-i", video_path,
        "-vn",          # drop the video stream
        "-ar", "16000", # resample to 16 kHz (required by the HF model)
        "-ac", "1",     # mono
        "-f", "wav",
        audio_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}: "
                f"{result.stderr.decode(errors='replace')}"
            )
        logger.info("Audio extracted to %s", audio_path)
        return audio_path
    except Exception:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise
