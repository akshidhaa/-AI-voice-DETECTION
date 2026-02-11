import os
import numpy as np
import soundfile as sf

# Create folders
os.makedirs("samples/human", exist_ok=True)
os.makedirs("samples/ai", exist_ok=True)

sr = 22050
duration = 3
t = np.linspace(0, duration, int(sr * duration), False)

# Human-like audio
for i in range(3):
    audio = np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(len(t))
    sf.write(f"samples/human/human_{i}.wav", audio, sr)

# AI-like audio
for i in range(3):
    audio = np.sin(2 * np.pi * 500 * t)
    sf.write(f"samples/ai/ai_{i}.wav", audio, sr)

print("DONE: audio files created")
