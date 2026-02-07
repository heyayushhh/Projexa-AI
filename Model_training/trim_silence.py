import os
import librosa
import soundfile as sf
from tqdm import tqdm

# ========================
# CONFIG
# ========================
INPUT_DIRS = {
    "clean_audio": "clean_audio",
    "stutter_audio": "stutter_audio"
}

OUTPUT_BASE = "trimmed_audio"

SAMPLE_RATE = 16000
TOP_DB = 30          # silence threshold
MIN_DURATION = 0.1  # seconds

# ========================
# TRIM FUNCTION
# ========================
def trim_audio(in_path, out_path):
    try:
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE)

        y_trimmed, _ = librosa.effects.trim(
            y,
            top_db=TOP_DB
        )

        if len(y_trimmed) < MIN_DURATION * sr:
            return False

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, y_trimmed, sr)
        return True

    except Exception as e:
        print(f"❌ Error: {in_path} | {e}")
        return False


# ========================
# PROCESS DATA
# ========================
for label, input_root in INPUT_DIRS.items():
    print(f"\n🔹 Processing {label}")

    for speaker in os.listdir(input_root):
        speaker_path = os.path.join(input_root, speaker)
        if not os.path.isdir(speaker_path):
            continue

        for session in os.listdir(speaker_path):
            session_path = os.path.join(speaker_path, session)
            if not os.path.isdir(session_path):
                continue

            wav_files = [
                f for f in os.listdir(session_path)
                if f.endswith(".wav")
            ]

            print(f"  ▶ {speaker}/{session} ({len(wav_files)} files)")

            for wav in tqdm(wav_files, leave=False):
                in_path = os.path.join(session_path, wav)

                out_path = os.path.join(
                    OUTPUT_BASE,
                    label,
                    speaker,
                    session,
                    wav
                )

                trim_audio(in_path, out_path)

print("\n✅ All audio trimmed successfully!")