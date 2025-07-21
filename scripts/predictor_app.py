import gradio as gr
import joblib
import pandas as pd
import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel
import os

# === Load model and scaler ===
model = joblib.load("models/voice_rfc_model.joblib")
scaler = joblib.load("models/voice_scaler.joblib")

# === Setup OpenSMILE extractor ===
smile = Smile(
    feature_set=FeatureSet.eGeMAPSv02,
    feature_level=FeatureLevel.Functionals
)

# === Inference function ===
def predict_health(audio):
    try:
        # Check if audio file was uploaded
        if audio is None:
            return {"Error": "Please upload an audio file"}
        
        # Extract features from uploaded .wav file
        # audio is already a filepath when using type="filepath"
        features = smile.process_file(audio)
        features_df = features.reset_index(drop=True)

        # Scale using pre-fitted scaler
        features_scaled = scaler.transform(features_df)

        # Predict
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)

        # Format the result
        status_emoji = "✅ Healthy" if prediction == "healthy" else "⚠️ COVID Positive"
        
        return {
            "Health Status": f"{status_emoji} ({prediction})",
            "Confidence Score": f"{confidence * 100:.2f}%",
            "Probabilities": f"Healthy: {proba[0]*100:.1f}% | Positive: {proba[1]*100:.1f}%"
        }

    except Exception as e:
        return { 
            "Error": f"Processing failed: {str(e)}",
            "Suggestion": "Please ensure you uploaded a valid WAV audio file with clear voice recording"
        }

# === Launch the UI ===
demo = gr.Interface(
    fn=predict_health,
    inputs=gr.Audio(type="filepath", label="🎤 Upload vowel-a.wav recording"),
    outputs=gr.JSON(label="🔍 Prediction Results"),
    title="🎤 VoiceVitals: AI Health Classifier",
    description="""
    **Upload a vowel 'A' sound recording** (e.g., 'ahhh' for 2-3 seconds) to predict health status.
    
    📊 **Model Info:**
    - Algorithm: Random Forest Classifier (81.36% accuracy)
    - Features: 88 voice biomarkers (OpenSMILE eGeMAPSv02)
    - Training: 295 voice samples from Coswara dataset
    
    🎯 **Best Results:** Clear, steady 'ahhh' sound for 2-3 seconds
    """,
    examples=[
        ["data/samples/recorded_voice.wav"] if os.path.exists("data/samples/recorded_voice.wav") else []
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
