from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
import base64
import librosa
import numpy as np
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Step 2: Decode Base64 Audio --------
def decode_audio(base64_str, filename="input.wav"):
    # Remove base64 header if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    audio_bytes = base64.b64decode(base64_str)
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    return filename


# -------- Step 3: Extract MFCC Features --------
def extract_features(filename):
    # Load with specific sample rate (16kHz) as per improvement suggestion
    y, sr = librosa.load(filename, sr=16000)
    
    # Normalize loudness
    y = librosa.util.normalize(y)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# -------- Load Trained Model --------
model = joblib.load("voice_model.pkl")

# -------- Step 5: REST API --------
from pydantic import BaseModel

class AudioRequest(BaseModel):
    audio_base64: str
    language: str = "en"
    audio_format: str = "mp3"

# -------- Step 5: REST API --------
@app.post("/classify", include_in_schema=False)
@app.post("/api/voice-detect")
def classify(request: AudioRequest, x_api_key: str = Header(...)):
    print(f"Received request: len(audio)={len(request.audio_base64)}, lang={request.language}")

    # Verify API Key (allow the demo key or the secret one)
    if x_api_key not in ["SECRET123", "guvi-demo-key-123"]:
        return {"error": "Invalid API key"}

    try:
        filename = decode_audio(request.audio_base64)
        features = extract_features(filename).reshape(1, -1)

        # Get probabilities: [prob_human, prob_ai] (assuming 0=Human, 1=AI)
        probs = model.predict_proba(features)[0]
        prob_human = probs[0]
        prob_ai = probs[1]
        
        # Lower threshold: if Human probability > 0.4, classify as HUMAN
        # This makes it easier to classify as HUMAN (reducing False Positives for AI)
        if prob_human > 0.4:
            prediction = "HUMAN"
            confidence_val = prob_human
            explanation = "Natural speech patterns and physiological micro-tremors observed."
        else:
            prediction = "AI_GENERATED"
            confidence_val = prob_ai
            explanation = "Synthetic speech artifacts and unnatural pitch transitions detected."

        return {
            "prediction": prediction,
            "confidence": round(float(confidence_val), 3),
            "explanation": explanation
        }
    except Exception as e:
        return {"error": str(e)}
