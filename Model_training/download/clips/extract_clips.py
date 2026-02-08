#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc.
#
"""
For each podcast episode:
* Load the full 16kHz WAV
* Extract labeled segments using Start / Stop (sample indices)
* Save each clip as a new WAV file

Output: Model_training/audio_clips/
"""

from pathlib import Path
import argparse
import pandas as pd
from scipy.io import wavfile

# -------------------- BASE DIR --------------------
# download/clips/extract_clips.py → Model_training
BASE_DIR = Path(__file__).resolve().parents[2]

# -------------------- ARGPARSE --------------------
parser = argparse.ArgumentParser(
    description="Extract labeled clips from SEP-28k or FluencyBank"
)

parser.add_argument(
    "--labels",
    type=str,
    default="dataset/SEP-28k_labels.csv",
    help="Labels CSV (relative to Model_training)"
)

parser.add_argument(
    "--progress",
    action="store_true",
    help="Show progress bar"
)

args = parser.parse_args()

# -------------------- PATHS --------------------
LABELS_CSV = BASE_DIR / args.labels
WAV_ROOT = BASE_DIR / "raw_audio"
CLIP_ROOT = BASE_DIR / "audio_clips"

CLIP_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------- HELPERS --------------------
def norm(s: str) -> str:
    """Normalize show names for matching"""
    return s.lower().replace(" ", "").strip()

def safe_name(s: str) -> str:
    """Make string safe for Windows filenames"""
    return (
        s.replace(" ", "")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
    )

# -------------------- VALIDATION --------------------
if not LABELS_CSV.exists():
    raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

if not WAV_ROOT.exists():
    raise FileNotFoundError(f"Raw audio directory not found: {WAV_ROOT}")

# -------------------- DISCOVER WAVS --------------------
available_shows = {
    norm(p.name): p.name
    for p in WAV_ROOT.iterdir()
    if p.is_dir()
}

# -------------------- LOAD LABELS --------------------
df = pd.read_csv(LABELS_CSV)

required_cols = {"Show", "EpId", "ClipId", "Start", "Stop"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------- ITERATION SETUP --------------------
indices = range(len(df))
if args.progress:
    from tqdm import tqdm
    indices = tqdm(indices, desc="Extracting clips")

loaded_wav_path = None
audio = None
sample_rate = None

# -------------------- PROCESS --------------------
for i in indices:
    row = df.iloc[i]

    raw_show = str(row["Show"])
    show_key = norm(raw_show)

    if show_key not in available_shows:
        continue

    show = available_shows[show_key]
    safe_show = safe_name(show)

    # Force clean ints
    episode = str(int(float(row["EpId"])))
    clip_id = str(int(float(row["ClipId"])))

    start = int(row["Start"])
    stop = int(row["Stop"])

    wav_path = WAV_ROOT / show / f"{episode}.wav"
    if not wav_path.exists():
        continue

    clip_dir = CLIP_ROOT / safe_show / episode
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_path = clip_dir / f"{safe_show}_{episode}_{clip_id}.wav"

    # Load WAV only when episode changes
    if wav_path != loaded_wav_path:
        sample_rate, audio = wavfile.read(wav_path)
        if sample_rate != 16000:
            raise ValueError(f"{wav_path} is not 16kHz")
        loaded_wav_path = wav_path

    clip_audio = audio[start:stop]
    wavfile.write(clip_path, sample_rate, clip_audio)

print("✅ Labeled clip extraction complete → Model_training/audio_clips/")

#run -> python download/clips/extract_clips.py --progress