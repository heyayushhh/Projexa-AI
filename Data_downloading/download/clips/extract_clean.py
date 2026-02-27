#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc.
#
"""
Extract ONLY clean (fluent) speech clips:
- Prolongation == 0
- Block == 0
- SoundRep == 0
- WordRep == 0
- DifficultToUnderstand == 0

Save them under Model_training/clean_audio/
"""

from pathlib import Path
import argparse
import pandas as pd
from scipy.io import wavfile

# -------------------- BASE DIR --------------------
# download/clips/extract_clean.py → Model_training
BASE_DIR = Path(__file__).resolve().parents[2]

# -------------------- ARGPARSE --------------------
parser = argparse.ArgumentParser(
    description="Extract ONLY clean speech clips from SEP-28k / FluencyBank"
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
OUT_ROOT = BASE_DIR / "clean_audio"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------- HELPERS --------------------
def norm(s: str) -> str:
    return s.lower().replace(" ", "").strip()

def safe_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "")
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

required_cols = {
    "Show", "EpId", "ClipId", "Start", "Stop",
    "Prolongation", "Block", "SoundRep",
    "WordRep", "DifficultToUnderstand"
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------- FILTER: CLEAN SPEECH ONLY --------------------
df = df[
    (df["Prolongation"] == 0) &
    (df["Block"] == 0) &
    (df["SoundRep"] == 0) &
    (df["WordRep"] == 0) &
    (df["DifficultToUnderstand"] == 0)
]

print(f"✅ Found {len(df)} clean speech clips")

# -------------------- ITERATION SETUP --------------------
indices = range(len(df))
if args.progress:
    from tqdm import tqdm
    indices = tqdm(indices, desc="Extracting clean clips")

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

    episode = str(int(float(row["EpId"])))
    clip_id = str(int(float(row["ClipId"])))

    start = int(row["Start"])
    stop = int(row["Stop"])

    wav_path = WAV_ROOT / show / f"{episode}.wav"
    if not wav_path.exists():
        continue

    clip_dir = OUT_ROOT / safe_show / episode
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

print("✅ Clean speech clips saved to Model_training/clean_audio/")

# run -> python download/clips/extract_clean.py --progress