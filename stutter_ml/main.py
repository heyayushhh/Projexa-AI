from model_loader import load_model
from audio_utils import load_audio, split_audio
from inference import run_inference
from postprocess import post_process

# 1. Load model
classifier = load_model()

# 2. Load audio (🔥 limit to 5 min for now)
audio, sr = load_audio("sample.wav", max_duration=300)

# DEBUG INFO
print("Audio duration (sec):", len(audio) / sr)

# 3. Split audio
chunks, timestamps = split_audio(audio, sr)

print("Total chunks:", len(chunks))

# 4. Run inference
raw_results = run_inference(classifier, chunks, timestamps)

# 5. Post-process
final_results = post_process(raw_results)

# 6. Output
print("\n🎯 FINAL RESULTS:\n")

if not final_results:
    print("No stutter detected")
else:
    for r in final_results:
        print(f"{r['start']:.2f}s - {r['end']:.2f}s | {r['label']} | {r['confidence']:.2f}")