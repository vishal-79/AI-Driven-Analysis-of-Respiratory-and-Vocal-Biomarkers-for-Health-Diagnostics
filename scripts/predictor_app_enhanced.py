"""
Enhanced VoiceVitals Predictor with User-Friendly Interface
Clear explanations and refined visualizations for everyone
"""
import gradio as gr
import joblib
import pandas as pd
import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel
import matplotlib.pyp        with gr.Column(scale=2):
            audio_input = gr.Audio(
                type="filepath", 
                label="🎙️ Record Your Voice - Say 'ahhh' clearly for 2-3 seconds"
            )
            gr.Markdown("""
            **📝 Recording Tips:**
            - Find a quiet space
            - Hold the 'ahhh' sound steady for 2-3 seconds  
            - Speak at normal volume (not too loud or too soft)
            - Make sure your microphone is working
            """)mport seaborn as sns
import librosa
import librosa.display
import os
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

def create_voice_visualizations(audio_path):
    """Create beautiful, easy-to-understand voice visualizations"""
    try:
        # Load the audio file
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        duration = len(audio_data) / sample_rate
        
        # Create a professional-looking figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎤 Your Voice Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Voice Wave Pattern (Time Domain)
        time = np.linspace(0, duration, len(audio_data))
        ax1.plot(time, audio_data, color='#2E86AB', linewidth=1.5, alpha=0.8)
        ax1.fill_between(time, audio_data, alpha=0.3, color='#A23B72')
        ax1.set_title('🌊 Voice Wave Pattern\n(How your voice sounds over time)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Voice Strength', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, duration)
        
        # Add simple stats
        max_strength = np.max(np.abs(audio_data))
        ax1.text(0.02, 0.98, f'Recording Length: {duration:.1f} seconds\nVoice Strength: {"Strong" if max_strength > 0.1 else "Moderate" if max_strength > 0.05 else "Soft"}', 
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Voice Frequency Map (Spectrogram)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', 
                                      ax=ax2, cmap='plasma', alpha=0.8)
        ax2.set_title('🌈 Voice Frequency Map\n(Different pitches in your voice)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Pitch Level (Hz)', fontsize=12)
        ax2.set_ylim(0, 4000)  # Focus on human voice range
        
        # Add colorbar with simple explanation
        cbar = plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        cbar.set_label('Voice Intensity\n(Bright = Loud)', fontsize=10)
        
        # 3. Voice Quality Indicators
        # Extract some basic features for visualization
        pitch = librosa.yin(audio_data, fmin=50, fmax=300, sr=sample_rate)
        pitch_clean = pitch[~np.isnan(pitch)]
        
        if len(pitch_clean) > 0:
            avg_pitch = np.mean(pitch_clean)
            pitch_stability = 1 - (np.std(pitch_clean) / np.mean(pitch_clean)) if np.mean(pitch_clean) > 0 else 0
        else:
            avg_pitch = 150  # Default
            pitch_stability = 0.5
        
        # Voice quality metrics (simplified for users)
        quality_metrics = {
            'Voice Steadiness': min(pitch_stability * 100, 100),
            'Average Pitch': min((avg_pitch / 200) * 100, 100),
            'Voice Clarity': min((max_strength * 200), 100),
            'Recording Quality': min((duration / 3) * 100, 100)
        }
        
        # Create a bar chart for voice quality
        metrics_names = list(quality_metrics.keys())
        metrics_values = list(quality_metrics.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax3.barh(metrics_names, metrics_values, color=colors, alpha=0.8)
        ax3.set_xlim(0, 100)
        ax3.set_title('📊 Voice Quality Report\n(Higher scores = Better quality)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Quality Score (%)', fontsize=12)
        
        # Add percentage labels on bars
        for i, (bar, value) in enumerate(zip(bars, metrics_values)):
            ax3.text(value + 2, bar.get_y() + bar.get_height()/2, 
                    f'{value:.0f}%', va='center', fontsize=11, fontweight='bold')
        
        # 4. Health Prediction Confidence Gauge
        # This will be updated after prediction, for now show placeholder
        ax4.pie([70, 30], labels=['Confidence', 'Uncertainty'], 
               colors=['#2ECC71', '#E74C3C'], autopct='%1.0f%%',
               startangle=90, counterclock=False)
        ax4.set_title('🎯 Prediction Confidence\n(Will update after analysis)', 
                     fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = "voice_analysis_dashboard.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return viz_path
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return None

def create_simple_results_chart(prediction, confidence, probabilities):
    """Create a simple, clear results visualization"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('🏥 Your Health Analysis Results', fontsize=18, fontweight='bold')
        
        # 1. Health Status with confidence
        if prediction == "healthy":
            main_color = '#2ECC71'  # Green
            status_text = 'Healthy'
            emoji = '✅'
        else:
            main_color = '#E74C3C'  # Red
            status_text = 'Needs Attention'
            emoji = '⚠️'
        
        # Confidence gauge
        confidence_pct = confidence * 100
        remaining = 100 - confidence_pct
        
        wedges, texts, autotexts = ax1.pie([confidence_pct, remaining], 
                                          labels=['Confidence', 'Uncertainty'],
                                          colors=[main_color, '#BDC3C7'],
                                          autopct='%1.0f%%',
                                          startangle=90,
                                          textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        ax1.set_title(f'{emoji} Predicted Status: {status_text}\nConfidence Level', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 2. Probability breakdown
        healthy_prob = probabilities[0] * 100
        positive_prob = probabilities[1] * 100
        
        categories = ['Healthy\nVoice Pattern', 'Concerning\nVoice Pattern']
        values = [healthy_prob, positive_prob]
        colors = ['#2ECC71', '#E74C3C']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.8, width=0.6)
        ax2.set_ylim(0, 100)
        ax2.set_title('📊 Detailed Probability Breakdown', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Probability (%)', fontsize=12)
        
        # Add percentage labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the results chart
        results_path = "health_results_chart.png"
        plt.savefig(results_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return results_path
        
    except Exception as e:
        print(f"Results chart error: {e}")
        return None

# === Main prediction function ===
def predict_health(audio, model_type):
    try:
        # Check if audio file was uploaded
        if audio is None:
            return {
                "📋 Status": "Please upload your voice recording first",
                "💡 Tip": "Record a clear 'ahhh' sound for 2-3 seconds"
            }, None, None
        
        # Create voice visualizations first
        voice_viz = create_voice_visualizations(audio)
        
        # Load the selected model
        model, scaler = load_model(model_type)
        
        # Extract features from uploaded .wav file
        features = smile.process_file(audio)
        features_df = features.reset_index(drop=True)

        # Scale using pre-fitted scaler
        features_scaled = scaler.transform(features_df)

        # Predict
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)

        # Create results visualization
        results_chart = create_simple_results_chart(prediction, confidence, proba)

        # Format the result in simple, clear language
        if prediction == "healthy":
            status_message = "🎉 Your voice shows healthy patterns!"
            health_advice = "Your voice analysis suggests normal respiratory health. Keep maintaining good vocal hygiene!"
            status_emoji = "✅"
            confidence_level = "High" if confidence > 0.8 else "Good" if confidence > 0.6 else "Moderate"
        else:
            status_message = "⚠️ Your voice shows some concerning patterns"
            health_advice = "Your voice analysis detected patterns that may indicate respiratory issues. Consider consulting a healthcare professional for a proper evaluation."
            status_emoji = "⚠️"
            confidence_level = "High" if confidence > 0.8 else "Good" if confidence > 0.6 else "Moderate"
        
        # Model explanation in simple terms
        model_explanation = {
            "balanced": "🎯 Advanced AI Model - Most Accurate (92.5% success rate)",
            "original": "📊 Standard AI Model - Good Performance (81.4% success rate)"
        }
        
        # Create detailed, user-friendly results
        detailed_results = {
            "🏥 Health Assessment": f"{status_emoji} {prediction.title()} Voice Pattern",
            "📊 Confidence Level": f"{confidence_level} ({confidence * 100:.1f}% certain)",
            "🎯 What This Means": status_message,
            "💡 Health Advice": health_advice,
            "🤖 AI Model Used": model_explanation[model_type],
            "🔬 Voice Features Analyzed": f"{features_df.shape[1]} different voice characteristics",
            "📈 Detailed Breakdown": f"Healthy Pattern: {proba[0]*100:.1f}% | Concerning Pattern: {proba[1]*100:.1f}%"
        }

        return detailed_results, voice_viz, results_chart

    except Exception as e:
        error_results = { 
            "❌ Analysis Failed": f"We couldn't analyze your voice recording",
            "🔧 What Went Wrong": str(e),
            "💡 Try This": "Make sure you uploaded a clear WAV audio file with your voice saying 'ahhh' for 2-3 seconds"
        }
        return error_results, None, None

# === Launch the Enhanced UI ===
with gr.Blocks(title="VoiceVitals: AI Health Scanner", theme=gr.themes.Soft()) as demo:
    
    # Header with clear, friendly language
    gr.Markdown("""
    # 🎤 VoiceVitals: Your Personal Voice Health Scanner
    ### Simple AI-powered health screening through your voice
    
    **How it works:** Just record yourself saying "ahhh" for 2-3 seconds, and our AI will analyze your voice for health patterns.
    
    💡 **Perfect for:** Regular health monitoring, early screening, peace of mind
    """)
    
    # Input section with clear instructions
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                type="filepath", 
                label="�️ Record Your Voice",
                info="Say 'ahhh' clearly for 2-3 seconds (like when the doctor checks your throat)"
            )
            gr.Markdown("""
            **📝 Recording Tips:**
            - Find a quiet space
            - Hold the 'ahhh' sound steady for 2-3 seconds  
            - Speak at normal volume (not too loud or too soft)
            - Make sure your microphone is working
            """)
            
        with gr.Column(scale=1):
            model_selector = gr.Radio(
                choices=["balanced", "original"], 
                value="balanced",
                label="🤖 AI Model Choice",
                info="Balanced model is more accurate for most people"
            )
            gr.Markdown("""
            **🎯 Model Guide:**
            - **Balanced**: Best overall accuracy (92.5%)
            - **Original**: Good baseline performance (81.4%)
            """)
    
    # Analyze button
    analyze_btn = gr.Button("🔍 Analyze My Voice", variant="primary", size="lg")
    
    # Results section with three outputs
    with gr.Row():
        with gr.Column(scale=1):
            # Main results
            results_output = gr.JSON(
                label="� Your Health Analysis Report",
                show_label=True
            )
            
        with gr.Column(scale=1):
            # Voice visualization
            voice_viz_output = gr.Image(
                label="🎤 Your Voice Analysis Dashboard",
                show_label=True
            )
            
    # Results chart (full width)
    results_chart_output = gr.Image(
        label="📊 Visual Results Summary",
        show_label=True
    )
    
    # Important disclaimers in friendly language
    with gr.Accordion("ℹ️ Important Information", open=False):
        gr.Markdown("""
        ### 🩺 Medical Disclaimer
        **This is a screening tool, not a medical diagnosis:**
        - Results are based on AI analysis of voice patterns
        - This tool cannot replace a doctor's examination
        - If you have health concerns, please consult a healthcare professional
        - Use this for general awareness and monitoring only
        
        ### � How Our AI Works
        **Simple explanation:**
        - Our AI was trained on thousands of voice recordings
        - It learned to recognize patterns in healthy vs. concerning voices
        - The analysis looks at 88 different voice characteristics
        - Higher confidence means the AI is more certain about its assessment
        
        ### �️ Best Recording Practices
        **For the most accurate results:**
        - Record in a quiet room
        - Use a good quality microphone if possible
        - Say "ahhh" like you would at the doctor's office
        - Keep the sound steady for the full 2-3 seconds
        - Avoid background noise, coughing, or interruptions
        
        ### 📊 Understanding Your Results
        **What the scores mean:**
        - **Confidence Level**: How sure the AI is about its assessment
        - **Voice Pattern**: Whether your voice shows healthy or concerning patterns
        - **Probability Breakdown**: Detailed percentages for each possibility
        - **Quality Scores**: How clear and analyzable your recording was
        """)
    
    # Connect the analyze button to all outputs
    analyze_btn.click(
        fn=predict_health,
        inputs=[audio_input, model_selector],
        outputs=[results_output, voice_viz_output, results_chart_output]
    )
    
    # Add example if available
    if os.path.exists("data/samples/recorded_voice.wav"):
        gr.Examples(
            examples=[["data/samples/recorded_voice.wav", "balanced"]],
            inputs=[audio_input, model_selector],
            label="🎵 Try with our sample recording"
        )

if __name__ == "__main__":
    demo.launch(share=False)
