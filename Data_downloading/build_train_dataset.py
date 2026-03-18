import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).resolve().parent

NORMALIZED_ROOT = BASE_DIR / "normalized_audio"
LABELS_CSV = BASE_DIR / "dataset" / "SEP-28k_labels.csv"
OUTPUT_CSV = BASE_DIR / "train_dataset.csv"

SAMPLE_RATE = 16000

# ===================== LABEL MAP =====================
LABEL_MAP = {
    "fluent": 0,
    "prolongation": 1,
    "block": 2,
    "sound_rep": 3,
    "word_rep": 4,
    "difficult": 5,
    "interjection": 6
}

COLUMN_TO_LABEL = {
    "Prolongation": "prolongation",
    "Block": "block",
    "SoundRep": "sound_rep",
    "WordRep": "word_rep",
    "DifficultToUnderstand": "difficult",
    "Interjection": "interjection"
}

# ===================== HELPERS =====================
def safe_name(s):
    return str(s).strip().replace(" ", "").replace("/", "_").replace("\\", "_")

def get_label(row):
    """
    Robust label assignment
    Priority:
    1. Any stutter type
    2. Else fluent (NoStutteredWords / NaturalPause)
    """

    # Collect all active stutters
    active = []

    for col, label_name in COLUMN_TO_LABEL.items():
        if row[col] > 0:
            active.append(label_name)

    # If any stutter exists → pick first (or customize priority)
    if active:
        return LABEL_MAP[active[0]]

    # Else fluent
    if row.get("NoStutteredWords", 0) > 0 or row.get("NaturalPause", 0) > 0:
        return LABEL_MAP["fluent"]

    return LABEL_MAP["fluent"]


# ===================== LOAD LABELS =====================
df_labels = pd.read_csv(LABELS_CSV)

label_lookup = {}

for _, row in df_labels.iterrows():
    show = safe_name(row["Show"])
    ep = str(int(float(row["EpId"])))
    clip = str(int(float(row["ClipId"])))

    key = f"{show}_{ep}_{clip}"

    label_lookup[key] = get_label(row)

print(f"✅ Loaded {len(label_lookup)} label mappings")

# ===================== BUILD DATASET =====================
rows = []

# 🔥 IMPORTANT: scan ALL wavs regardless of folder
all_wavs = list(NORMALIZED_ROOT.rglob("*.wav"))

print(f"\n🔹 Found {len(all_wavs)} audio files")

for wav_path in tqdm(all_wavs):

    try:
        filename = wav_path.stem

        if filename not in label_lookup:
            continue

        label = label_lookup[filename]

        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        duration = len(y) / sr

        if duration < 0.3:
            continue

        rows.append({
            "path": wav_path.as_posix(),
            "label": label,
            "label_name": list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(label)],
            "duration": round(duration, 3)
        })

    except Exception as e:
        print(f"❌ {wav_path} | {e}")

# ===================== SAVE =====================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Dataset created: {OUTPUT_CSV}")
print(f"Total samples: {len(df)}")

# ===================== STATS =====================
print("\n📊 Class distribution:")
print(df["label_name"].value_counts())

# 🔥 Optional: show imbalance ratio
counts = Counter(df["label"])
print("\n⚖️ Imbalance ratio:")
for k, v in counts.items():
    print(f"{k}: {v}")