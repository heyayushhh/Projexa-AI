import librosa

# Load audio
def load_audio(file_path, max_duration=None):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # 🔥 Limit long audio (IMPORTANT)
    if max_duration:
        audio = audio[:int(max_duration * sr)]

    return audio, sr


# ✅ FIXED chunking (robust for long audio)
def split_audio(audio, sr, chunk_duration=1.5, stride=0.5):
    chunk_size = int(chunk_duration * sr)
    stride_size = int(stride * sr)

    chunks = []
    timestamps = []

    i = 0
    while i < len(audio):
        end_i = i + chunk_size

        chunk = audio[i:end_i]

        start = i / sr
        end = min(end_i, len(audio)) / sr

        chunks.append(chunk)
        timestamps.append((start, end))

        i += stride_size

    return chunks, timestamps