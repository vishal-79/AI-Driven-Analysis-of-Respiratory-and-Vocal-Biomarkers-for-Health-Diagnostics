"""
Test the balanced model with proper feature alignment
"""
import joblib
import pandas as pd
from opensmile import Smile, FeatureSet, FeatureLevel
import numpy as np

def test_balanced_model():
    """Test the balanced model with a sample audio file"""
    
    # Load the balanced model
    model = joblib.load("models/voice_rfc_model_balanced.joblib")
    scaler = joblib.load("models/voice_scaler_balanced.joblib")
    
    # Initialize OpenSMILE
    smile = Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.Functionals
    )
    
    # Test with sample audio
    test_audio = "data/samples/recorded_voice.wav"
    
    if not os.path.exists(test_audio):
        print(f"❌ Test audio not found: {test_audio}")
        return
    
    print("🧪 Testing Balanced Model:")
    print(f"📁 Audio file: {test_audio}")
    
    try:
        # Extract features
        features = smile.process_file(test_audio)
        features_df = features.reset_index(drop=True)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        print(f"\n🔍 Balanced Model Results:")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence*100:.1f}%")
        print(f"   Probabilities: Healthy={proba[0]*100:.1f}%, Positive={proba[1]*100:.1f}%")
        
        # Compare with original model
        model_orig = joblib.load("models/voice_rfc_model.joblib")
        scaler_orig = joblib.load("models/voice_scaler.joblib")
        
        features_scaled_orig = scaler_orig.transform(features_df)
        prediction_orig = model_orig.predict(features_scaled_orig)[0]
        proba_orig = model_orig.predict_proba(features_scaled_orig)[0]
        
        print(f"\n📊 Original Model Results:")
        print(f"   Prediction: {prediction_orig}")
        print(f"   Confidence: {np.max(proba_orig)*100:.1f}%")
        print(f"   Probabilities: Healthy={proba_orig[0]*100:.1f}%, Positive={proba_orig[1]*100:.1f}%")
        
        print(f"\n🎯 Comparison:")
        print(f"   Same prediction: {'✅' if prediction == prediction_orig else '❌'}")
        print(f"   Confidence difference: {(confidence - np.max(proba_orig))*100:+.1f} percentage points")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import os
    test_balanced_model()
