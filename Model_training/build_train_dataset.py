import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

# -------------------- CONFIG --------------------
TRIMMED_ROOT = Path("normalized_audio")
OUTPUT_CSV = "train_dataset.csv"

LABEL_MAP = {
    "clean_audio": 0,
    "stutter_clips": 1   # ✅ fixed here
}

SAMPLE_RATE = 16000

rows = []

# -------------------- SCAN DATASET --------------------
for label_folder, label_value in LABEL_MAP.items():
    base_dir = TRIMMED_ROOT / label_folder

    if not base_dir.exists():
        print(f"⚠️ Skipping missing folder: {base_dir}")
        continue

    print(f"\n🔹 Processing {label_folder}")

    for show_dir in base_dir.iterdir():
        if not show_dir.is_dir():
            continue

        for ep_dir in show_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            wav_files = list(ep_dir.glob("*.wav"))

            for wav_path in tqdm(wav_files, leave=False):
                try:
                    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
                    duration = len(y) / sr

                    # Skip extremely short clips
                    if duration < 0.2:
                        continue

                    rows.append({
                        "path": wav_path.as_posix(),
                        "label": label_value,
                        "duration": round(duration, 3)
                    })

                except Exception as e:
                    print(f"❌ Failed: {wav_path} | {e}")

# -------------------- SAVE --------------------
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Training dataset created: {OUTPUT_CSV}")
print(df.head())
print(f"\nTotal samples: {len(df)}")
print("\nClass distribution:")
print(df['label'].value_counts().sort_index())