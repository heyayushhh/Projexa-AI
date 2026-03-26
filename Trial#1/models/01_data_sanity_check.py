import pandas as pd
from pathlib import Path
from collections import Counter
import librosa

# =============================
# PATHS
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Data_downloading" / "dataset"
RAW_AUDIO_DIR = BASE_DIR / "Data_downloading" / "raw_audio"

LABELS_CSV = DATASET_DIR / "SEP-28k_labels.csv"

# =============================
# LOAD LABELS
# =============================
df = pd.read_csv(LABELS_CSV)

print("\n✅ Total rows in labels:", len(df))

# =============================
# TARGET LABELS
# =============================
TARGET_LABELS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
    "NoStutteredWords"
]

# =============================
# CHECK MULTI-LABEL
# =============================
def count_active_labels(row):
    return sum([row[label] > 0 for label in TARGET_LABELS])

df["num_active_labels"] = df.apply(count_active_labels, axis=1)

print("\n📊 Label Type Distribution:")
print(df["num_active_labels"].value_counts())

# =============================
# FILTER VALID CLIPS (>=1 label)
# =============================
valid_df = df[df["num_active_labels"] > 0].copy()

print("\n✅ Valid labeled clips:", len(valid_df))

# =============================
# CLASS DISTRIBUTION
# =============================
label_counts = {}

for label in TARGET_LABELS:
    label_counts[label] = (df[label] > 0).sum()

print("\n📊 Class Distribution:")
for k, v in label_counts.items():
    print(f"{k}: {v}")

# =============================
# CHECK AUDIO FILES
# =============================
missing_files = 0
duration_stats = []

for idx, row in valid_df.iterrows():
    show = row["Show"]
    ep_id = row["EpId"]
    clip_id = row["ClipId"]

    file_path = RAW_AUDIO_DIR / show / f"{ep_id}_{clip_id}.wav"

    if not file_path.exists():
        missing_files += 1
        continue

    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        duration_stats.append(duration)
    except:
        continue

print("\n❌ Missing files:", missing_files)

if duration_stats:
    print("\n⏱ Duration stats:")
    print("Min:", min(duration_stats))
    print("Max:", max(duration_stats))
    print("Avg:", sum(duration_stats) / len(duration_stats))