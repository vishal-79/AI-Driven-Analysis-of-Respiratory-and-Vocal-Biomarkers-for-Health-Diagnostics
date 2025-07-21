# 🎤 VoiceVitals Health Predictor App

## Overview
The VoiceVitals app is an AI-powered voice analysis tool that can predict potential COVID-19 status from voice recordings. It uses advanced machine learning techniques trained on the Coswara dataset.

## How to Use

### 1. Web Interface (Gradio)
```bash
# Start the web app
python scripts/predictor_app.py

# Then open: http://127.0.0.1:7860
```

### 2. Programmatic Usage
```python
from scripts.predictor_app import predict_health

# Test with an audio file
result = predict_health("path/to/audio.wav")
print(result)
```

## Recording Guidelines

### ✅ For Best Results:
- **Duration**: 2-3 seconds of steady 'ahhh' sound
- **Format**: WAV file format preferred
- **Quality**: Clear recording without background noise
- **Microphone**: Close to mouth (6-12 inches)
- **Environment**: Quiet room

### 🎯 Recording Instructions:
1. Take a deep breath
2. Say 'ahhh' in a natural, comfortable tone
3. Hold the sound steady for 2-3 seconds
4. Avoid voice breaks or variations

## Model Information

### 📊 Performance Metrics:
- **Overall Accuracy**: 81.36%
- **Sensitivity (COVID Detection)**: 95.0%
- **Specificity (Healthy Detection)**: 52.6%
- **Algorithm**: Random Forest Classifier
- **Features**: 88 voice biomarkers

### 🔬 Key Voice Features:
1. **Spectral Slopes** - Voice quality indicators
2. **Loudness Dynamics** - Voice intensity patterns
3. **Formant Frequencies** - Vocal tract characteristics
4. **MFCC Features** - Spectral envelope properties
5. **Jitter & Shimmer** - Voice stability measures

### ⚠️ Important Notes:
- This is a research tool, not a medical diagnostic device
- Results should not replace professional medical advice
- The model is trained on specific dataset conditions
- Individual voice variations may affect accuracy

## Technical Details

### Dependencies:
- `gradio` - Web interface
- `opensmile` - Feature extraction
- `scikit-learn` - Machine learning
- `pandas`, `numpy` - Data processing

### Files:
- `models/voice_rfc_model.joblib` - Trained model
- `models/voice_scaler.joblib` - Feature scaler
- `scripts/predictor_app.py` - Web application
- `scripts/test_predictor.py` - Testing script

### Feature Extraction:
The app uses OpenSMILE's eGeMAPSv02 feature set, which includes:
- Fundamental frequency (F0) statistics
- Loudness and spectral features
- Voice quality measures (jitter, shimmer, HNR)
- Formant frequencies and bandwidths
- Cepstral coefficients (MFCCs)

## Troubleshooting

### Common Issues:
1. **"No module named 'gradio'"** → Install: `pip install gradio`
2. **"Model file not found"** → Run training script first
3. **"Audio processing error"** → Check file format (WAV preferred)
4. **Low confidence scores** → Try clearer recording

### Audio Format Support:
- Primary: WAV files
- Secondary: MP3, M4A (may work but not optimal)
- Sample Rate: Any (OpenSMILE handles conversion)
- Channels: Mono or Stereo

## Example Usage

```python
# Load and test the predictor
from scripts.predictor_app import predict_health

# Test with sample audio
result = predict_health("data/samples/recorded_voice.wav")

# Expected output:
{
    "Health Status": "✅ Healthy (healthy)",
    "Confidence Score": "61.00%",
    "Probabilities": "Healthy: 61.0% | Positive: 39.0%"
}
```

## Disclaimer
This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult healthcare professionals for medical concerns.
