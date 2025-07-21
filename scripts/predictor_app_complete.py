"""
VoiceVitals: Enhanced AI Health Classifier with SHAP Explanations
Features: Model selection, SHAP explanations, feature importance charts, and detailed analysis
"""
import gradio as gr
import joblib
import pandas as pd
import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
import librosa
import librosa.display
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive use
plt.switch_backend('Agg')

# === Setup OpenSMILE extractor ===
smile = Smile(
    feature_set=FeatureSet.eGeMAPSv02,
    feature_level=FeatureLevel.Functionals
)

def load_model(model_type="balanced"):
    """Load the specified model type"""
    if model_type == "balanced":
        model = joblib.load("models/voice_rfc_model_balanced.joblib")
        scaler = joblib.load("models/voice_scaler_balanced.joblib")
    else:
        model = joblib.load("models/voice_rfc_model.joblib")
        scaler = joblib.load("models/voice_scaler.joblib")
    return model, scaler

def create_feature_importance_plot(model, feature_names, model_type):
    """Create a matplotlib plot of top 10 feature importances"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'🔍 Top 10 Most Important Voice Features\n{model_type.title()} Model', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(indices))
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    bars = plt.barh(y_pos, importances[indices], color=colors, alpha=0.8)
    
    # Customize the plot
    plt.yticks(y_pos, [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.ylabel('Voice Features', fontsize=12)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.grid(axis='x', alpha=0.3)
    
    # Save and return path
    plot_path = f"temp_feature_importance_{model_type}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path

def create_shap_explanation(model, features_scaled, feature_names, model_type):
    """Create SHAP explanation plot"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_scaled)
        
        # For binary classification, take the positive class SHAP values
        if len(shap_values) == 2:
            shap_values_plot = shap_values[1]  # Positive class
        else:
            shap_values_plot = shap_values
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_plot, features_scaled, 
                         feature_names=feature_names, 
                         plot_type="bar", max_display=15, show=False)
        
        plt.title(f'🎯 SHAP Feature Impact Analysis\n{model_type.title()} Model - Current Prediction', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Mean |SHAP Value| (Impact on Prediction)', fontsize=12)
        plt.tight_layout()
        
        # Save and return path
        shap_path = f"temp_shap_{model_type}.png"
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return shap_path, shap_values_plot
    
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None, None

def create_audio_visualizations(audio_path):
    """Create waveform and spectrogram visualizations"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. Time-domain waveform
        time = np.linspace(0, len(y) / sr, len(y))
        ax1.plot(time, y, color='steelblue', linewidth=0.8)
        ax1.set_title('📈 Time-Domain Waveform\nAmplitude vs Time', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(y) / sr)
        
        # Add some statistics
        duration = len(y) / sr
        max_amplitude = np.max(np.abs(y))
        ax1.text(0.02, 0.95, f'Duration: {duration:.2f}s\nMax Amplitude: {max_amplitude:.3f}\nSample Rate: {sr} Hz', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='viridis')
        ax2.set_title('🌈 Spectrogram\nFrequency vs Time (Color = Amplitude)', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Frequency (Hz)', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2, format='%+2.0f dB')
        cbar.set_label('Amplitude (dB)', fontsize=10)
        
        plt.tight_layout()
        
        # Save and return path
        viz_path = "temp_audio_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return viz_path
        
    except Exception as e:
        print(f"Audio visualization failed: {e}")
        return None

def create_feature_table(features_df, feature_names, shap_values=None):
    """Create a detailed feature table with values and SHAP impacts"""
    feature_data = []
    
    for i, feature_name in enumerate(feature_names):
        # Convert feature value to Python float to avoid numpy formatting issues
        try:
            feature_value = float(features_df.iloc[0, i]) if i < len(features_df.columns) else 0.0
        except (TypeError, ValueError):
            feature_value = 0.0
        
        # Handle SHAP values properly
        if shap_values is not None:
            try:
                # Extract scalar value from SHAP array - handle different array structures
                if hasattr(shap_values, '__len__') and len(shap_values) > 0:
                    if hasattr(shap_values[0], '__len__') and len(shap_values[0]) > i:
                        shap_val = shap_values[0][i]
                        # Convert to scalar if it's an array
                        if hasattr(shap_val, 'item'):
                            shap_impact = float(shap_val.item())
                        elif hasattr(shap_val, '__len__') and len(shap_val) == 1:
                            shap_impact = float(shap_val[0])
                        else:
                            shap_impact = float(shap_val)
                    else:
                        shap_impact = 0.0
                else:
                    shap_impact = 0.0
            except (TypeError, IndexError, ValueError):
                shap_impact = 0.0
        else:
            shap_impact = 0.0
        
        # Categorize features
        if 'F0' in feature_name or 'frequency' in feature_name.lower():
            category = "🎵 Pitch & Frequency"
        elif 'loudness' in feature_name.lower():
            category = "🔊 Loudness & Intensity"
        elif 'mfcc' in feature_name.lower():
            category = "🌊 Spectral Features (MFCC)"
        elif 'jitter' in feature_name.lower() or 'shimmer' in feature_name.lower():
            category = "📈 Voice Quality"
        elif 'slope' in feature_name.lower():
            category = "📊 Spectral Slopes"
        elif 'bandwidth' in feature_name.lower():
            category = "📐 Bandwidth Features"
        else:
            category = "🔧 Other Features"
        
        feature_data.append({
            'Rank': i + 1,
            'Feature Name': feature_name,
            'Category': category,
            'Value': f"{feature_value:.4f}",
            'SHAP Impact': f"{shap_impact:.4f}" if shap_values is not None else "N/A",
            'Impact Direction': "→ Positive" if shap_impact > 0 else "→ Negative" if shap_impact < 0 else "→ Neutral" if shap_values is not None else "N/A"
        })
    
    return pd.DataFrame(feature_data)

# === Main prediction function ===
def predict_health_with_explanation(audio, model_type):
    """Main prediction function with SHAP explanations"""
    try:
        if audio is None:
            return {}, None, None, None, pd.DataFrame()
        
        # Create audio visualizations first
        audio_viz = create_audio_visualizations(audio)
        
        # Load model and extract features
        model, scaler = load_model(model_type)
        features = smile.process_file(audio)
        features_df = features.reset_index(drop=True)
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        # Convert numpy arrays to Python floats to avoid formatting issues
        proba_healthy = float(proba[0])
        proba_positive = float(proba[1])
        confidence_float = float(confidence)
        
        # Create visualizations
        feature_names = features_df.columns.tolist()
        importance_plot = create_feature_importance_plot(model, feature_names, model_type)
        shap_plot, shap_values = create_shap_explanation(model, features_scaled, feature_names, model_type)
        
        # Create feature table
        feature_table = create_feature_table(features_df, feature_names, shap_values)
        
        # Format results with proper float conversion
        status_emoji = "✅ Healthy" if prediction == "healthy" else "⚠️ COVID Positive"
        model_info = {
            "balanced": "🎯 Balanced Model (SMOTE - 92.5% accuracy)",
            "original": "📊 Original Model (81.4% accuracy)"
        }
        
        results = {
            "🏥 Health Status": f"{status_emoji} ({prediction})",
            "🎯 Confidence Score": f"{confidence_float * 100:.1f}%",
            "📊 Probability Breakdown": f"Healthy: {proba_healthy*100:.1f}% | COVID-Positive: {proba_positive*100:.1f}%",
            "🤖 Model Information": model_info[model_type],
            "🔬 Features Analyzed": f"{features_df.shape[1]} voice biomarkers",
            "📈 Prediction Strength": "High Confidence" if confidence_float > 0.8 else "Medium Confidence" if confidence_float > 0.6 else "Low Confidence"
        }
        
        return results, audio_viz, importance_plot, shap_plot, feature_table
        
    except Exception as e:
        error_results = {
            "❌ Error": f"Processing failed: {str(e)}",
            "💡 Suggestion": "Please ensure you uploaded a valid WAV audio file with clear voice recording"
        }
        return error_results, None, None, None, pd.DataFrame()

# === Individual tab functions ===
def prediction_tab(audio, model_type):
    """Main prediction tab"""
    results, _, _, _, _ = predict_health_with_explanation(audio, model_type)
    return results

def audio_preview_tab(audio, model_type):
    """Audio preview visualization tab"""
    _, audio_viz, _, _, _ = predict_health_with_explanation(audio, model_type)
    return audio_viz

def feature_importance_tab(audio, model_type):
    """Feature importance visualization tab"""
    _, _, importance_plot, _, _ = predict_health_with_explanation(audio, model_type)
    return importance_plot

def shap_explanation_tab(audio, model_type):
    """SHAP explanation tab"""
    _, _, _, shap_plot, _ = predict_health_with_explanation(audio, model_type)
    return shap_plot

def feature_table_tab(audio, model_type):
    """Feature table tab"""
    _, _, _, _, feature_table = predict_health_with_explanation(audio, model_type)
    return feature_table

# === Create the Gradio interface ===
def create_interface():
    """Create the main Gradio interface with tabs"""
    
    with gr.Blocks(title="VoiceVitals: AI Health Classifier", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🎤 VoiceVitals: Enhanced AI Health Classifier
        ### Advanced voice-based health screening with explainable AI
        
        Upload a vowel 'A' sound recording (2-3 seconds of steady 'ahhh') for AI-powered health analysis.
        """)
        
        # Input controls
        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    type="filepath", 
                    label="🎤 Upload Voice Recording (vowel-a.wav)"
                )
                gr.Markdown("*Record or upload a clear 'ahhh' sound (2-3 seconds)*")
            with gr.Column(scale=1):
                model_selector = gr.Radio(
                    choices=["balanced", "original"], 
                    value="balanced",
                    label="🤖 Model Selection"
                )
                gr.Markdown("*Balanced model recommended for better accuracy*")
        
        # Prediction button
        predict_btn = gr.Button("🔍 Analyze Voice", variant="primary", size="lg")
        
        # Audio Preview Section (appears as soon as audio is uploaded)
        with gr.Group():
            gr.Markdown("## 🎧 Audio Preview")
            audio_viz_output = gr.Image(
                label="📊 Waveform & Spectrogram Analysis",
                show_label=True,
                visible=False
            )
            audio_preview_info = gr.Markdown("""
            **📈 Audio Analysis:**
            - **Waveform**: Shows amplitude changes over time
            - **Spectrogram**: Displays frequency content with color-coded intensity
            - **Statistics**: Duration, amplitude range, and sample rate information
            """, visible=False)
        
        # Results tabs
        with gr.Tabs():
            
            # Tab 1: Main Prediction Results
            with gr.TabItem("🏥 Health Prediction"):
                prediction_output = gr.JSON(
                    label="📊 Analysis Results",
                    show_label=True
                )
                gr.Markdown("""
                **📝 How to interpret results:**
                - **Health Status**: Primary prediction (Healthy vs COVID-Positive)
                - **Confidence Score**: Model's certainty in the prediction
                - **Probability Breakdown**: Detailed probability for each class
                - **Prediction Strength**: Qualitative assessment of confidence level
                """)
            
            # Tab 2: Feature Importance Chart
            with gr.TabItem("📊 Feature Importance"):
                importance_plot_output = gr.Image(
                    label="🔍 Top 10 Most Important Voice Features",
                    show_label=True
                )
                gr.Markdown("""
                **📈 Understanding Feature Importance:**
                - Higher bars indicate features that contribute more to the model's decisions
                - Features are ranked by their overall impact across all predictions
                - Voice biomarkers include pitch, loudness, spectral characteristics, and voice quality measures
                """)
            
            # Tab 3: SHAP Explanations
            with gr.TabItem("🎯 SHAP Analysis"):
                shap_plot_output = gr.Image(
                    label="🧠 SHAP Feature Impact for Current Prediction",
                    show_label=True
                )
                gr.Markdown("""
                **🎯 SHAP (SHapley Additive exPlanations):**
                - Shows how each feature specifically influenced THIS prediction
                - Positive values push toward the predicted class
                - Negative values push toward the other class
                - Provides personalized explanations for each voice sample
                """)
            
            # Tab 4: Full Feature Table
            with gr.TabItem("📋 Detailed Features"):
                feature_table_output = gr.DataFrame(
                    label="🔬 Complete Feature Analysis (88 Voice Biomarkers)",
                    show_label=True,
                    interactive=False,
                    wrap=True
                )
                gr.Markdown("""
                **📋 Complete Feature Breakdown:**
                - **Feature Name**: Technical name of the voice biomarker
                - **Category**: Grouped by type (Pitch, Loudness, Spectral, etc.)
                - **Value**: Extracted value from your voice recording
                - **SHAP Impact**: How this feature influenced the current prediction
                - **Impact Direction**: Whether the feature pushed toward Healthy or COVID-Positive
                """)
        
        # Model information
        with gr.Accordion("ℹ️ Model Information & Technical Details", open=False):
            gr.Markdown("""
            ### 🤖 Model Specifications:
            - **Algorithm**: Random Forest Classifier
            - **Features**: 88 voice biomarkers (OpenSMILE eGeMAPSv02)
            - **Training Data**: Coswara dataset with voice recordings
            - **Balanced Model**: 92.5% accuracy (SMOTE-enhanced)
            - **Original Model**: 81.4% accuracy (baseline)
            
            ### 🎙️ Recording Guidelines:
            - **Duration**: 2-3 seconds of steady vowel sound
            - **Sound**: Clear 'ahhh' (like saying 'car' without the 'cr')
            - **Environment**: Quiet room, minimal background noise
            - **Distance**: 6-12 inches from microphone
            - **Format**: WAV files preferred, MP3 acceptable
            
            ### ⚠️ Important Disclaimers:
            - This is a research tool for educational purposes
            - Not intended for medical diagnosis or treatment decisions
            - Results should not replace professional healthcare advice
            - Individual voice variations may affect accuracy
            """)
        
        # Connect button to all outputs
        predict_btn.click(
            fn=lambda audio, model: predict_health_with_explanation(audio, model),
            inputs=[audio_input, model_selector],
            outputs=[prediction_output, audio_viz_output, importance_plot_output, shap_plot_output, feature_table_output]
        )
        
        # Show audio preview immediately when audio is uploaded
        def show_audio_preview(audio):
            if audio is not None:
                viz = create_audio_visualizations(audio)
                return gr.update(value=viz, visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False), gr.update(visible=False)
        
        audio_input.change(
            fn=show_audio_preview,
            inputs=[audio_input],
            outputs=[audio_viz_output, audio_preview_info]
        )
        
        # Add examples if available
        if os.path.exists("data/samples/recorded_voice.wav"):
            gr.Examples(
                examples=[["data/samples/recorded_voice.wav", "balanced"]],
                inputs=[audio_input, model_selector],
                label="🎵 Try with sample audio"
            )
    
    return demo

# === Launch the application ===
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7870,
        show_error=True
    )
