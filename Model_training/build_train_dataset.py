import pandas as pd
from pathlib import Path

# -------------------- CONFIG --------------------
LABELS_CSV = "SEP-28k_labels.csv"
STUTTER_DIR = Path("stutter_audio")
CLEAN_DIR = Path("clean_audio")
OUTPUT_CSV = "train_dataset.csv"

# -------------------- LOAD LABELS --------------------
df = pd.read_csv(LABELS_CSV)

# Normalize numeric columns
df["EpId"] = df["EpId"].astype(int)
df["ClipId"] = df["ClipId"].astype(int)

# Index labels by (Show, EpId, ClipId)
label_index = {
    (row.Show.strip(), row.EpId, row.ClipId): (row.Start, row.Stop)
    for _, row in df.iterrows()
}

rows = []

# -------------------- HELPER --------------------
def process_root(root_dir: Path, stutter_label: int):
    fold1 = root_dir.name  # stutter_audio / clean_audio

    for show_dir in root_dir.iterdir():
        if not show_dir.is_dir():
            continue

        fold2 = show_dir.name  # HeStutters

        for ep_dir in show_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            fold3 = ep_dir.name  # episode id

            for wav in ep_dir.glob("*.wav"):
                # HeStutters_1_0.wav → ClipId = 0
                clip_id = int(wav.stem.split("_")[-1])

                key = (fold2, int(fold3), clip_id)
                if key not in label_index:
                    continue

                start, end = label_index[key]

                rows.append({
                    "file_name": wav.name,
                    "fold1": fold1,
                    "fold2": fold2,
                    "fold3": fold3,
                    "start": start,
                    "end": end,
                    "stutter": stutter_label
                })

# -------------------- BUILD DATASET --------------------
process_root(STUTTER_DIR, stutter_label=1)
process_root(CLEAN_DIR, stutter_label=0)

# -------------------- SAVE --------------------
train_df = pd.DataFrame(rows)
train_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Training dataset created: {OUTPUT_CSV}")
print(train_df.head())