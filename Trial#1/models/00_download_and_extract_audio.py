import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests

# ==============================
# PATH SETUP
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "Data_downloading" / "dataset"
RAW_AUDIO_DIR = BASE_DIR / "Data_downloading" / "raw_audio"

EPISODES_CSV = DATASET_DIR / "SEP-28k_episodes.csv"
LABELS_CSV = DATASET_DIR / "SEP-28k_labels.csv"

# ==============================
# LOAD DATA
# ==============================

episodes_df = pd.read_csv(
    EPISODES_CSV,
    header=None,
    names=["Title", "EpisodeName", "URL", "Show", "EpId"]
)

labels_df = pd.read_csv(LABELS_CSV)

# ==============================
# 🔥 CLEAN DATA (IMPORTANT)
# ==============================

# Strip spaces
labels_df["Show"] = labels_df["Show"].astype(str).str.strip()
episodes_df["Show"] = episodes_df["Show"].astype(str).str.strip()

# Convert EpId to int
labels_df["EpId"] = labels_df["EpId"].astype(int)
episodes_df["EpId"] = episodes_df["EpId"].astype(int)

# ==============================
# MERGE
# ==============================

df = labels_df.merge(
    episodes_df,
    on=["Show", "EpId"],
    how="left"
)

# ==============================
# DEBUG CHECK
# ==============================

print("Missing URLs:", df["URL"].isna().sum())

# ==============================
# CREATE DIR
# ==============================

RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# DOWNLOAD MP3
# ==============================

def download_mp3(url, output_path):
    if pd.isna(url):
        return False

    if output_path.exists():
        return True

    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code != 200:
            return False

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return True
    except:
        return False

# ==============================
# CUT AUDIO
# ==============================

def cut_audio(input_file, start, end, output_file):
    duration = (end - start) / 16000

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_file),
        "-ss", str(start / 16000),
        "-t", str(duration),
        "-ac", "1",
        "-ar", "16000",
        str(output_file)
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==============================
# PIPELINE
# ==============================

print("Starting pipeline...")

for show in df["Show"].unique():
    show_dir = RAW_AUDIO_DIR / show
    show_dir.mkdir(parents=True, exist_ok=True)

    show_df = df[df["Show"] == show]

    for ep_id in tqdm(show_df["EpId"].unique(), desc=show):

        ep_df = show_df[show_df["EpId"] == ep_id]
        url = ep_df.iloc[0]["URL"]

        if pd.isna(url):
            print(f"Skipping {show} Ep {ep_id} (no URL)")
            continue

        temp_mp3 = show_dir / f"{ep_id}.mp3"
        temp_wav = show_dir / f"{ep_id}_full.wav"

        # Download MP3
        success = download_mp3(url, temp_mp3)

        if not success:
            print(f"Failed download: {url}")
            continue

        # Convert to WAV
        if not temp_wav.exists():
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(temp_mp3),
                "-ac", "1",
                "-ar", "16000",
                str(temp_wav)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Cut clips
        for _, row in ep_df.iterrows():
            clip_id = row["ClipId"]
            start = row["Start"]
            stop = row["Stop"]

            output_file = show_dir / f"{ep_id}_{clip_id}.wav"

            if output_file.exists():
                continue

            cut_audio(temp_wav, start, stop, output_file)

print("DONE ✅")