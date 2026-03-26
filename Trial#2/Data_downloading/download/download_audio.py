import pathlib
import subprocess
import argparse
import pandas as pd
import requests
import librosa
import soundfile as sf
import numpy as np
from scipy.io import wavfile

# ===================== BASE =====================
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

# ===================== ARGPARSE =====================
parser = argparse.ArgumentParser(description="Full SEP-28k pipeline")

parser.add_argument("--episodes", type=str, default="dataset/SEP-28k_episodes.csv")
parser.add_argument("--labels", type=str, default="dataset/SEP-28k_labels.csv")
parser.add_argument("--progress", action="store_true")

args = parser.parse_args()

# ===================== PATHS =====================
RAW_AUDIO = BASE_DIR / "raw_audio"
CLEAN_AUDIO = BASE_DIR / "clean_audio"
STUTTER_AUDIO = BASE_DIR / "stutter_clips"
NORMALIZED_AUDIO = BASE_DIR / "normalized_audio"

RAW_AUDIO.mkdir(exist_ok=True)
CLEAN_AUDIO.mkdir(exist_ok=True)
STUTTER_AUDIO.mkdir(exist_ok=True)
NORMALIZED_AUDIO.mkdir(exist_ok=True)

# ===================== CONFIG =====================
AUDIO_EXTS = [".mp3", ".m4a", ".mp4"]
CHUNK_SIZE = 8192
SAMPLE_RATE = 16000
TOP_DB = 30
MIN_DURATION = 0.1
TARGET_RMS = 0.05
EPS = 1e-8

# ===================== HELPERS =====================
def safe_name(s):
    return str(s).strip().replace(" ", "").replace("/", "_").replace("\\", "_")

def norm(s):
    return s.lower().replace(" ", "").strip()

# ===================== DOWNLOAD =====================
def download_audio(url, output_path):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return True
    except:
        return False

def download_and_convert():
    print("\n⬇️ Downloading audio...")

    df = pd.read_csv(BASE_DIR / args.episodes)

    iterator = range(len(df))
    if args.progress:
        from tqdm import tqdm
        iterator = tqdm(iterator)

    for i in iterator:
        row = df.iloc[i]

        url = row.iloc[2]
        show = safe_name(row.iloc[-2])
        ep = str(int(float(row.iloc[-1])))

        ext = next((e for e in AUDIO_EXTS if e in url), None)
        if not ext:
            continue

        out_dir = RAW_AUDIO / show
        out_dir.mkdir(parents=True, exist_ok=True)

        orig = out_dir / f"{ep}{ext}"
        wav = out_dir / f"{ep}.wav"

        if wav.exists():
            continue

        if not orig.exists():
            if not download_audio(url, orig):
                continue

        subprocess.run([
            "ffmpeg", "-y", "-i", str(orig),
            "-ac", "1", "-ar", "16000", str(wav)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        orig.unlink()

    print("✅ Download complete")

# ===================== CLIP EXTRACTION =====================
def extract_clips():
    print("\n✂️ Extracting clips...")

    df = pd.read_csv(BASE_DIR / args.labels)

    available = {
        norm(p.name): p.name
        for p in RAW_AUDIO.iterdir() if p.is_dir()
    }

    iterator = range(len(df))
    if args.progress:
        from tqdm import tqdm
        iterator = tqdm(iterator)

    loaded_path = None
    audio = None

    for i in iterator:
        row = df.iloc[i]

        show_key = norm(row["Show"])
        if show_key not in available:
            continue

        show = available[show_key]
        safe_show = safe_name(show)

        ep = str(int(float(row["EpId"])))
        clip = str(int(float(row["ClipId"])))

        start, stop = int(row["Start"]), int(row["Stop"])

        wav_path = RAW_AUDIO / show / f"{ep}.wav"
        if not wav_path.exists():
            continue

        if wav_path != loaded_path:
            sr, audio = wavfile.read(wav_path)
            loaded_path = wav_path

        clip_audio = audio[start:stop]

        # CLEAN
        if (
            row["Prolongation"] == 0 and
            row["Block"] == 0 and
            row["SoundRep"] == 0 and
            row["WordRep"] == 0 and
            row["DifficultToUnderstand"] == 0
        ):
            out = CLEAN_AUDIO / safe_show / ep
        else:
            out = STUTTER_AUDIO / safe_show / ep

        out.mkdir(parents=True, exist_ok=True)

        out_path = out / f"{safe_show}_{ep}_{clip}.wav"
        wavfile.write(out_path, sr, clip_audio)

    print("✅ Clip extraction done")

# ===================== NORMALIZATION =====================
def trim_and_normalize(in_path, out_path):
    try:
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE)

        y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB)

        if len(y_trim) < MIN_DURATION * sr:
            return

        rms = np.sqrt(np.mean(y_trim ** 2))
        y_norm = y_trim * (TARGET_RMS / (rms + EPS))
        y_norm = np.clip(y_norm, -1, 1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y_norm, sr)

    except:
        pass

def normalize_all():
    print("\n🔊 Normalizing audio...")

    for label, folder in {
        "clean_audio": CLEAN_AUDIO,
        "stutter_clips": STUTTER_AUDIO
    }.items():

        if not folder.exists():
            continue

        for speaker in folder.iterdir():
            for session in speaker.iterdir():
                for wav in session.glob("*.wav"):

                    out = NORMALIZED_AUDIO / label / speaker.name / session.name / wav.name
                    trim_and_normalize(wav, out)

    print("✅ Normalization done")

# ===================== MAIN =====================
if __name__ == "__main__":
    download_and_convert()
    extract_clips()
    normalize_all()

    print("\n🚀 FULL PIPELINE COMPLETE")