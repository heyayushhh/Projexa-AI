import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -------------------- CONFIG --------------------
INPUT_ROOT = Path("trimmed_audio")
OUTPUT_ROOT = Path("normalized_audio")

TARGET_RMS = 0.05
SAMPLE_RATE = 16000
EPS = 1e-8

LABEL_FOLDERS = ["clean_audio", "stutter_audio"]

# -------------------- NORMALIZATION --------------------
def rms_normalize(y, target_rms=TARGET_RMS):
    rms = np.sqrt(np.mean(y ** 2))
    gain = target_rms / (rms + EPS)
    y_norm = y * gain

    # prevent clipping
    y_norm = np.clip(y_norm, -1.0, 1.0)
    return y_norm

# -------------------- PROCESS DATA --------------------
for label in LABEL_FOLDERS:
    in_base = INPUT_ROOT / label
    out_base = OUTPUT_ROOT / label

    if not in_base.exists():
        print(f"⚠️ Missing folder: {in_base}")
        continue

    print(f"\n🔹 Normalizing {label}")

    for show_dir in in_base.iterdir():
        if not show_dir.is_dir():
            continue

        for ep_dir in show_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            wav_files = list(ep_dir.glob("*.wav"))

            for wav_path in tqdm(wav_files, leave=False):
                try:
                    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

                    if len(y) == 0:
                        continue

                    y_norm = rms_normalize(y)

                    out_path = out_base / show_dir.name / ep_dir.name / wav_path.name
                    os.makedirs(out_path.parent, exist_ok=True)

                    sf.write(out_path, y_norm, sr)

                except Exception as e:
                    print(f"❌ Failed: {wav_path} | {e}")

print("\n✅ Loudness normalization complete!")