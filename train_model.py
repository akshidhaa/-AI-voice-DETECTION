import os
import librosa
import numpy as np
import joblib
from sklearn.svm import SVC

def extract_features(filename):
    y, sr = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

# Load HUMAN voices (label = 0)
for file in os.listdir("samples/human"):
    path = os.path.join("samples/human", file)
    features = extract_features(path)
    X.append(features)
    y.append(0)

# Load AI voices (label = 1)
for file in os.listdir("samples/ai"):
    path = os.path.join("samples/ai", file)
    features = extract_features(path)
    X.append(features)
    y.append(1)

X = np.array(X)
y = np.array(y)

model = SVC(probability=True)
model.fit(X, y)

joblib.dump(model, "voice_model.pkl")

print("✅ Model trained and saved as voice_model.pkl")
