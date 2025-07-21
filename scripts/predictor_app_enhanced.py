"""
Enhanced VoiceVitals Predictor with User-Friendly Interface
Clear explanations and refined visualizations for everyone
"""
import gradio as gr
import joblib
import pandas as pd
import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('default')
sns.set_palette("husl")

# === Setup OpenSMILE extractor ===
smile = Smile(
    feature_set=FeatureSet.eGeMAPSv02,
    feature_level=FeatureLevel.Functionals
)

def classify_risk(probability):
    """Map predicted probability to risk levels with visual indicators"""
    if probability < 0.3:
        return {
            "level": "Low Risk",
            "emoji": "🟢",
            "color": "#2ECC71",
            "description": "Voice patterns suggest normal respiratory health",
            "recommendation": "Continue maintaining good vocal hygiene and healthy habits"
        }
    elif probability < 0.7:
        return {
            "level": "Moderate Risk", 
            "emoji": "🟡",
            "color": "#F39C12",
            "description": "Some concerning voice patterns detected",
            "recommendation": "Monitor symptoms and consider consulting healthcare provider if persistent"
        }
    else:
        return {
            "level": "High Risk",
            "emoji": "🔴", 
            "color": "#E74C3C",
            "description": "Voice patterns suggest potential respiratory issues",
            "recommendation": "Strongly recommend consultation with healthcare professional for proper evaluation"
        }

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

def create_simple_results_chart(prediction, confidence, probabilities, risk_info):
    """Create a simple, clear results visualization with risk levels"""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('🏥 Your Health Analysis Results', fontsize=18, fontweight='bold')
        
        # 1. Risk Level Gauge
        risk_level = risk_info["level"]
        risk_emoji = risk_info["emoji"]
        risk_color = risk_info["color"]
        
        # Create risk level pie chart
        if "Low" in risk_level:
            risk_pct = 100 - (probabilities[1] * 100)
            colors = [risk_color, '#BDC3C7']
            labels = ['Healthy Pattern', 'Uncertainty']
        elif "Moderate" in risk_level:
            risk_pct = probabilities[1] * 100
            colors = [risk_color, '#BDC3C7']
            labels = ['Moderate Risk', 'Low Risk']
        else:  # High Risk
            risk_pct = probabilities[1] * 100
            colors = [risk_color, '#BDC3C7']
            labels = ['High Risk', 'Other']
        
        remaining = 100 - risk_pct
        wedges, texts, autotexts = ax1.pie([risk_pct, remaining], 
                                          labels=labels,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        ax1.set_title(f'{risk_emoji} Risk Assessment\n{risk_level}', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 2. Probability Score Display
        unhealthy_prob = probabilities[1] * 100
        healthy_prob = probabilities[0] * 100
        
        categories = ['Healthy\nPattern', 'Concerning\nPattern']
        values = [healthy_prob, unhealthy_prob]
        colors = ['#2ECC71', '#E74C3C']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.8, width=0.6)
        ax2.set_ylim(0, 100)
        ax2.set_title(f'📊 Prediction Score: {unhealthy_prob:.2f}%\nDetailed Probability Breakdown', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Probability (%)', fontsize=12)
        
        # Add percentage labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        # 3. Risk Level Visual Guide
        risk_levels = ['Low Risk\n(0-30%)', 'Moderate Risk\n(30-70%)', 'High Risk\n(70-100%)']
        risk_colors = ['#2ECC71', '#F39C12', '#E74C3C']
        current_score = unhealthy_prob
        
        # Create horizontal bar showing where current score falls
        y_pos = [0, 1, 2]
        bars = ax3.barh(y_pos, [30, 40, 30], left=[0, 30, 70], 
                       color=risk_colors, alpha=0.7, height=0.6)
        
        # Add current score marker
        ax3.axvline(x=current_score, color='black', linewidth=3, linestyle='--')
        ax3.text(current_score, 2.5, f'Your Score\n{current_score:.1f}%', 
                ha='center', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(risk_levels)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel('Risk Score (%)', fontsize=12)
        ax3.set_title('🎯 Risk Level Guide\n(Where You Stand)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add grid for easier reading
        ax3.grid(True, alpha=0.3, axis='x')
        
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
        
        # Get the probability of unhealthy pattern (assuming [0]=healthy, [1]=unhealthy)
        unhealthy_probability = proba[1]
        
        # Classify risk level based on probability
        risk_info = classify_risk(unhealthy_probability)

        # Create results visualization with risk information
        results_chart = create_simple_results_chart(prediction, confidence, proba, risk_info)

        # Format the result with risk-based messaging
        risk_level = risk_info["level"]
        risk_emoji = risk_info["emoji"] 
        risk_description = risk_info["description"]
        risk_recommendation = risk_info["recommendation"]
        
        # Create status message based on risk level
        if "Low" in risk_level:
            status_message = f"🎉 {risk_emoji} {risk_level} - Your voice shows healthy patterns!"
            health_advice = risk_recommendation
            status_emoji = "✅"
        elif "Moderate" in risk_level:
            status_message = f"⚠️ {risk_emoji} {risk_level} - Some patterns need attention"
            health_advice = risk_recommendation
            status_emoji = "🟡"
        else:  # High Risk
            status_message = f"🚨 {risk_emoji} {risk_level} - Important patterns detected"
            health_advice = risk_recommendation
            status_emoji = "🔴"
            
        confidence_level = "High" if confidence > 0.8 else "Good" if confidence > 0.6 else "Moderate"
        
        # Model explanation in simple terms
        model_explanation = {
            "balanced": "🎯 Advanced AI Model - Most Accurate (92.5% success rate)",
            "original": "📊 Standard AI Model - Good Performance (81.4% success rate)"
        }
        
        # Create detailed, user-friendly results with risk information
        detailed_results = {
            "🩺 Prediction Score": f"{unhealthy_probability:.3f} ({unhealthy_probability*100:.1f}%)",
            "🎯 Risk Assessment": f"{risk_emoji} {risk_level}",
            "📊 Confidence Level": f"{confidence_level} ({confidence * 100:.1f}% certain)",
            "🔍 What This Means": risk_description,
            "💡 Health Recommendation": risk_recommendation,
            "🤖 AI Model Used": model_explanation[model_type],
            "🔬 Voice Features Analyzed": f"{features_df.shape[1]} different voice characteristics",
            "📈 Detailed Breakdown": f"Healthy Pattern: {proba[0]*100:.1f}% | Concerning Pattern: {proba[1]*100:.1f}%",
            "⚕️ Medical Note": "This is a screening tool - consult healthcare provider for medical diagnosis"
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
    # 🎤 VoiceVitals: AI-Powered Voice Health Risk Scanner
    ### Advanced health screening with intelligent risk assessment
    
    **How it works:** Record yourself saying "ahhh" for 2-3 seconds, and our AI will analyze your voice patterns and provide a risk-based health assessment with color-coded results.
    
    💡 **Perfect for:** Regular health monitoring, early screening, risk assessment, peace of mind
    
    🎯 **Risk Levels:** 🟢 Low Risk (0-30%) | 🟡 Moderate Risk (30-70%) | 🔴 High Risk (70-100%)
    """)
    
    # Input section with clear instructions
    with gr.Row():
        with gr.Column(scale=2):
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
            """)
            
        with gr.Column(scale=1):
            model_selector = gr.Radio(
                choices=["balanced", "original"], 
                value="balanced",
                label="🤖 AI Model Choice"
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
                label="🎯 Risk-Based Health Analysis Report",
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
        
        ### 🔬 How Our AI Works
        **Simple explanation:**
        - Our AI was trained on thousands of voice recordings
        - It learned to recognize patterns in healthy vs. concerning voices
        - The analysis looks at 88 different voice characteristics
        - Higher confidence means the AI is more certain about its assessment
        
        ### 🎙️ Best Recording Practices
        **For the most accurate results:**
        - Record in a quiet room
        - Use a good quality microphone if possible
        - Say "ahhh" like you would at the doctor's office
        - Keep the sound steady for the full 2-3 seconds
        - Avoid background noise, coughing, or interruptions
        
        ### 📊 Understanding Your Results
        **What the scores mean:**
        - **Prediction Score**: Raw probability (0.000-1.000) of concerning patterns
        - **Risk Assessment**: Color-coded risk level based on probability thresholds
          - 🟢 **Low Risk** (0-30%): Voice patterns suggest normal health
          - 🟡 **Moderate Risk** (30-70%): Some concerning patterns detected  
          - 🔴 **High Risk** (70-100%): Significant patterns suggest medical consultation
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
