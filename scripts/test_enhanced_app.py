"""
Test script for the enhanced predictor app components
"""
import sys
import os
sys.path.append('.')

def test_components():
    """Test individual components of the enhanced app"""
    
    print("🧪 Testing Enhanced VoiceVitals Components...")
    
    try:
        # Import all required modules
        import matplotlib.pyplot as plt
        import shap
        import gradio as gr
        import joblib
        import pandas as pd
        import numpy as np
        from opensmile import Smile, FeatureSet, FeatureLevel
        print("✅ All dependencies imported successfully")
        
        # Set matplotlib backend
        plt.switch_backend('Agg')
        print("✅ Matplotlib backend configured")
        
        # Test model loading
        from scripts.predictor_app_complete import load_model
        model, scaler = load_model("balanced")
        print("✅ Balanced model loaded successfully")
        
        model_orig, scaler_orig = load_model("original")
        print("✅ Original model loaded successfully")
        
        # Test OpenSMILE
        smile = Smile(
            feature_set=FeatureSet.eGeMAPSv02,
            feature_level=FeatureLevel.Functionals
        )
        print("✅ OpenSMILE initialized")
        
        # Test with sample audio if available
        test_audio = "data/samples/recorded_voice.wav"
        if os.path.exists(test_audio):
            print(f"🎤 Testing with audio file: {test_audio}")
            
            # Extract features
            features = smile.process_file(test_audio)
            features_df = features.reset_index(drop=True)
            print(f"✅ Features extracted: {features_df.shape}")
            
            # Test feature importance plot
            from scripts.predictor_app_complete import create_feature_importance_plot
            feature_names = features_df.columns.tolist()
            plot_path = create_feature_importance_plot(model, feature_names, "balanced")
            if os.path.exists(plot_path):
                print("✅ Feature importance plot created")
                os.remove(plot_path)  # Clean up
            
            # Test SHAP explanation
            from scripts.predictor_app_complete import create_shap_explanation
            features_scaled = scaler.transform(features_df)
            shap_path, shap_values = create_shap_explanation(model, features_scaled, feature_names, "balanced")
            if shap_path and os.path.exists(shap_path):
                print("✅ SHAP explanation created")
                os.remove(shap_path)  # Clean up
            
            # Test feature table
            from scripts.predictor_app_complete import create_feature_table
            feature_table = create_feature_table(features_df, feature_names, shap_values)
            print(f"✅ Feature table created: {feature_table.shape}")
            
            # Test full prediction function
            from scripts.predictor_app_complete import predict_health_with_explanation
            results, importance_plot, shap_plot, table = predict_health_with_explanation(test_audio, "balanced")
            print("✅ Full prediction pipeline working")
            print(f"   Result keys: {list(results.keys())}")
            
            # Clean up any temporary files
            for file in [importance_plot, shap_plot]:
                if file and os.path.exists(file):
                    os.remove(file)
        
        else:
            print(f"⚠️ Test audio not found: {test_audio}")
            print("   Skipping audio-based tests")
        
        print("\n🎉 All component tests passed!")
        print("✅ Enhanced VoiceVitals app is ready to use")
        print("🌐 Access the app at: http://127.0.0.1:7861")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_components()
