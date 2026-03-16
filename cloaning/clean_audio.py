import os
from pydub import AudioSegment

# Input folder containing raw recordings
input_folder = "raw_audio"

# Output folder for training
output_folder = "training_audio"

os.makedirs(output_folder, exist_ok=True)

# Supported formats
audio_formats = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

# Chunk length (10 seconds)
chunk_length = 10 * 1000  # milliseconds

count = 1

for file in os.listdir(input_folder):
    if file.lower().endswith(audio_formats):

        input_path = os.path.join(input_folder, file)

        # Load audio
        audio = AudioSegment.from_file(input_path)

        # Convert to mono + 22050Hz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(22050)

        # Split into chunks
        for i in range(0, len(audio), chunk_length):
            chunk = audio[i:i + chunk_length]

            # Skip very short chunks
            if len(chunk) < 3000:
                continue

            output_path = os.path.join(output_folder, f"{count:04d}.wav")

            # Export chunk
            chunk.export(output_path, format="wav")

            print(f"Saved chunk: {file} -> {output_path}")

            count += 1

print("All audio converted, cleaned, and split into 10-second training clips.")
