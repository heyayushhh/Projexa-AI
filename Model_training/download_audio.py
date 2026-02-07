#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc.
#
"""
For each podcast episode:
* Download the raw mp3/m4a file
* Convert it to a 16kHz mono WAV file
* Remove the original file
* Skip broken / 404 URLs safely
"""

import pathlib
import subprocess
import argparse
import pandas as pd
import requests

# -------------------- ARGPARSE --------------------
parser = argparse.ArgumentParser(
    description="Download raw audio files and convert to 16kHz mono WAVs."
)

parser.add_argument(
    "--episodes",
    type=str,
    required=True,
    help="Path to the episodes CSV file (e.g., SEP-28k_episodes.csv)"
)

parser.add_argument(
    "--wavs",
    type=str,
    default="wavs",
    help="Output directory for WAV files"
)

parser.add_argument(
    "--progress",
    action="store_true",
    help="Show progress bar"
)

args = parser.parse_args()
episode_csv = args.episodes
wav_root = pathlib.Path(args.wavs)

# -------------------- CONFIG --------------------
AUDIO_EXTS = [".mp3", ".m4a", ".mp4"]
CHUNK_SIZE = 8192
REQUEST_TIMEOUT = 30

# -------------------- HELPERS --------------------
def safe_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

def download_audio(url: str, output_path: pathlib.Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False

# -------------------- LOAD CSV --------------------
df = pd.read_csv(episode_csv, skipinitialspace=True)

# -------------------- ITERATION SETUP --------------------
indices = range(len(df))
if args.progress:
    from tqdm import tqdm
    indices = tqdm(indices, desc="Downloading episodes")

# -------------------- PROCESS --------------------
for i in indices:
    row = df.iloc[i]

    episode_url = row.iloc[2]     # audio URL
    raw_show = row.iloc[-2]       # show abbreviation
    raw_ep_idx = row.iloc[-1]     # episode index

    show_abbrev = safe_name(raw_show)
    ep_idx = str(int(float(raw_ep_idx)))

    # Detect extension
    ext = None
    for e in AUDIO_EXTS:
        if e in episode_url:
            ext = e
            break

    if ext is None:
        continue

    # Output paths
    episode_dir = wav_root / show_abbrev
    episode_dir.mkdir(parents=True, exist_ok=True)

    audio_orig = episode_dir / f"{ep_idx}{ext}"
    wav_path = episode_dir / f"{ep_idx}.wav"

    # Skip if WAV already exists
    if wav_path.exists():
        continue

    # Download
    if not audio_orig.exists():
        success = download_audio(episode_url, audio_orig)
        if not success:
            continue

    # Convert
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_orig),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Cleanup
    audio_orig.unlink()

print("✅ Audio download & conversion complete (with progress bar).")


# run
# python download_audio.py --episodes SEP-28k_episodes.csv --wavs wavs --progress