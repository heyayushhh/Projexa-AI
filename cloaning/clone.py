import os
from TTS.api import TTS

# Folder containing processed training clips
speaker_folder = "training_audio"

# Collect all wav reference files
speaker_wavs = [
    os.path.join(speaker_folder, f)
    for f in os.listdir(speaker_folder)
    if f.endswith(".wav")
]

# Sort for consistency
speaker_wavs.sort()

print("Using reference clips:")
for wav in speaker_wavs:
    print(wav)

# Load XTTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Text you want the cloned voice to speak
text = """
hi my name is Nikhil
"""

# Generate cloned audio
tts.tts_to_file(
    text=text,
    speaker_wav=speaker_wavs,
    language="en",
    temperature=0.65,
    repetition_penalty=2.0,
    file_path="cloned_output.wav"
)

print("Voice cloning complete! Output saved as cloned_output.wav")
