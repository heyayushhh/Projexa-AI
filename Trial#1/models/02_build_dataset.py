import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =============================
# PATHS
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Data_downloading" / "dataset"
RAW_AUDIO_DIR = BASE_DIR / "Data_downloading" / "raw_audio"

LABELS_CSV = DATASET_DIR / "SEP-28k_labels.csv"
OUTPUT_CSV = DATASET_DIR  / "final_dataset.csv"

# =============================
# TARGET LABELS (NO FLUENT)
# =============================
TARGET_LABELS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection"
]

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(LABELS_CSV)

print("Total rows:", len(df))

# =============================
# PROCESS DATA
# =============================
data = []

missing = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    show = row["Show"]
    ep_id = row["EpId"]
    clip_id = row["ClipId"]

    file_path = RAW_AUDIO_DIR / show / f"{ep_id}_{clip_id}.wav"

    if not file_path.exists():
        missing += 1
        continue

    # Build multi-label vector
    labels = [int(row[label] > 0) for label in TARGET_LABELS]

    # Skip if no stutter present
    if sum(labels) == 0:
        continue

    data.append({
        "file_path": str(file_path),
        "Prolongation": labels[0],
        "Block": labels[1],
        "SoundRep": labels[2],
        "WordRep": labels[3],
        "Interjection": labels[4],
    })

print("\nMissing skipped:", missing)
print("Final samples:", len(data))

# =============================
# SAVE
# =============================
final_df = pd.DataFrame(data)
final_df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Saved to:", OUTPUT_CSV)