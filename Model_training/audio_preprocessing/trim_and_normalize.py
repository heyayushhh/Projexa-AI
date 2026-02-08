import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -------------------- BASE DIR --------------------
# audio_preprocessing/trim_and_normalize.py → Model_training
BASE_DIR = Path(__file__).resolve().parents[1]

# ========================
# CONFIG
# ========================
INPUT_DIRS = {
    "clean_audio": BASE_DIR / "clean_audio",
    "stutter_clips": BASE_DIR / "stutter_clips",
}

OUTPUT_BASE = BASE_DIR / "normalized_audio"

SAMPLE_RATE = 16000
TOP_DB = 30          # silence threshold (dB)
MIN_DURATION = 0.1  # seconds

TARGET_RMS = 0.05
EPS = 1e-8

# ========================
# TRIM + NORMALIZE
# ========================
def trim_and_normalize(in_path: Path, out_path: Path) -> bool:
    try:
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE)

        if len(y) == 0:
            return False

        # ✂️ Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)

        if len(y_trimmed) < MIN_DURATION * sr:
            return False

        # 🔊 RMS normalize
        rms = np.sqrt(np.mean(y_trimmed ** 2))
        gain = TARGET_RMS / (rms + EPS)
        y_norm = y_trimmed * gain

        # prevent clipping
        y_norm = np.clip(y_norm, -1.0, 1.0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y_norm, sr)
        return True

    except Exception as e:
        print(f"❌ Failed: {in_path} | {e}")
        return False

# ========================
# PROCESS DATA
# ========================
for label, input_root in INPUT_DIRS.items():
    if not input_root.exists():
        print(f"⚠️ Skipping missing folder: {input_root}")
        continue

    print(f"\n✂️🔊 Processing {label}")

    for speaker_dir in input_root.iterdir():
        if not speaker_dir.is_dir():
            continue

        for session_dir in speaker_dir.iterdir():
            if not session_dir.is_dir():
                continue

            wav_files = list(session_dir.glob("*.wav"))
            print(f"  ▶ {speaker_dir.name}/{session_dir.name} ({len(wav_files)} files)")

            for wav_path in tqdm(wav_files, leave=False):
                out_path = (
                    OUTPUT_BASE
                    / label
                    / speaker_dir.name
                    / session_dir.name
                    / wav_path.name
                )

                trim_and_normalize(wav_path, out_path)

print("\n✅ Trim + normalization complete → Model_training/normalized_audio/")

# run -> python audio_preprocessing/trim_and_normalize.py