import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from opensmile import Smile, FeatureSet, FeatureLevel

# === CONFIG ===
DURATION = 5  # seconds
SAMPLE_RATE = 22050
AUDIO_PATH = 'data/samples/recorded_voice.wav'
FEATURE_CSV_PATH = 'data/features/recorded_features.csv'

# === STEP 1: Record Voice ===
print("🎙️ Recording... Speak Now!")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()

os.makedirs(os.path.dirname(AUDIO_PATH), exist_ok=True)
write(AUDIO_PATH, SAMPLE_RATE, recording)
print(f"✅ Recording saved to {AUDIO_PATH}")

# === STEP 2: Load Audio & Extract MFCC ===
y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# === Optional: Estimate Pitch ===
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch = pitches[magnitudes == magnitudes.max()]
pitch = pitch[0] if len(pitch) > 0 else 0
print(f"🎯 Estimated Pitch: {pitch:.2f} Hz")

# === STEP 3: Plot MFCCs ===
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.tight_layout()
plt.show()

# === STEP 4: Extract OpenSMILE Features ===
print("🧠 Extracting OpenSMILE features...")

smile = Smile(
    feature_set=FeatureSet.eGeMAPSv02,
    feature_level=FeatureLevel.Functionals
)

features = smile.process_file(AUDIO_PATH)
print("🔬 Feature Snapshot:")
print(features.T.head())

# === STEP 5: Save Features to CSV ===
os.makedirs(os.path.dirname(FEATURE_CSV_PATH), exist_ok=True)
features.to_csv(FEATURE_CSV_PATH)
print(f"📁 Features saved to {FEATURE_CSV_PATH}")
