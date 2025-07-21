"""
Test script for the voice health predictor
"""
import sys
import os
sys.path.append('.')

from scripts.predictor_app import predict_health

def test_predictor():
    """Test the predictor with the sample audio file"""
    
    # Test with existing sample
    test_audio = "data/samples/recorded_voice.wav"
    
    if os.path.exists(test_audio):
        print("🧪 Testing VoiceVitals Predictor...")
        print(f"📁 Audio file: {test_audio}")
        
        result = predict_health(test_audio)
        
        print("\n🔍 Prediction Results:")
        for key, value in result.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Test audio file not found: {test_audio}")
        print("   Please ensure you have a recorded voice sample.")

if __name__ == "__main__":
    test_predictor()
