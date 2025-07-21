# 🎤 VoiceVitals: AI-Driven Voice Health Analysis System

**Advanced respiratory and vocal biomarker analysis for health diagnostics using machine learning and explainable AI.**

VoiceVitals is a comprehensive voice health analysis platform that uses cutting-edge AI to detect potential health conditions through voice pattern analysis. Built with OpenSMILE feature extraction, Random Forest classification, and explainable AI (SHAP), this system provides accurate health screening through simple voice recordings.

## 🌟 Key Features

- **🤖 Dual AI Models**: Original (81.4%) and SMOTE-enhanced balanced model (92.5% accuracy)
- **🎨 Beautiful Visualizations**: Real-time waveforms, spectrograms, and voice quality metrics
- **🧠 Explainable AI**: SHAP analysis showing exactly how the AI makes predictions
- **🌐 User-Friendly Interface**: Web-based app with non-technical language
- **📊 Comprehensive Analysis**: 88 voice biomarkers extracted per sample
- **⚡ Real-time Processing**: Instant results with professional visualizations

## 🎯 What Makes VoiceVitals Special

- **Medical-Grade Accuracy**: 92.5% classification accuracy on balanced datasets
- **Explainable Results**: Users understand WHY the AI made its prediction
- **Simple Usage**: Just say "ahhh" for 2-3 seconds to get comprehensive health insights
- **Professional Visualizations**: Publication-ready charts and analysis dashboards
- **Multiple Interfaces**: From basic prediction to complete analysis with SHAP explanations

## 📁 Project Structure

```
VoiceVitals/
├── 📂 scripts/                 # Core ML Pipeline
│   ├── 🏗️ build_dataset.py     # Dataset creation from Coswara data
│   ├── 🤖 train_model.py        # Original model training
│   ├── ⚖️ train_model_balanced.py # SMOTE-enhanced balanced training
│   ├── 🎯 predictor_app.py      # Basic prediction interface
│   ├── 🔬 predictor_app_complete.py # Full app with SHAP & visualizations
│   ├── ✨ predictor_app_refined.py  # User-friendly polished interface
│   └── 📊 compare_models.py     # Model performance comparison
│
├── 📂 models/                  # Trained AI Models
│   ├── 🎯 voice_rfc_model.joblib         # Original Random Forest model
│   ├── ⚖️ voice_rfc_model_balanced.joblib # SMOTE-enhanced model  
│   ├── 📏 voice_scaler.joblib             # Feature scaler (original)
│   └── 📏 voice_scaler_balanced.joblib    # Feature scaler (balanced)
│
├── 📂 data/                    # Data Storage
│   ├── 📂 samples/             # Example audio recordings
│   │   └── 🎵 recorded_voice.wav
│   └── 📂 features/            # Extracted feature datasets
│       ├── 📊 recorded_features.csv    # Individual recordings
│       └── 📈 voice_dataset.csv        # Complete training dataset
│
├── 📂 datasets/                # Training Data (excluded from Git)
│   └── 📂 Coswara/             # COVID voice dataset (ignored - too large)
│
├── 📂 notebooks/               # Research & Analysis
│   └── 🔬 feature_extraction.ipynb    # Experimental analysis
│
├── 📋 README.md               # This comprehensive guide
├── 🚫 .gitignore             # Git exclusions (datasets, temp files)
└── 📜 requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13.2)
- Windows/macOS/Linux compatible
- Microphone access for voice recording

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/vishal-79/AI-Driven-Analysis-of-Respiratory-and-Vocal-Biomarkers-for-Health-Diagnostics.git
cd VoiceVitals

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch VoiceVitals App

Choose your preferred interface:

#### 🎯 **Refined App** (Recommended - User-Friendly)
```bash
python scripts/predictor_app_refined.py
```
- Beautiful visualizations with plain English explanations
- Voice quality dashboard with real-time metrics
- Clear health recommendations and confidence scoring

#### 🔬 **Complete App** (Advanced - For Researchers)
```bash
python scripts/predictor_app_complete.py
```
- Full SHAP explainable AI analysis
- Feature importance rankings
- Complete 88-feature breakdown with categories
- Technical analysis for researchers

#### ⚡ **Basic App** (Simple - Quick Predictions)
```bash
python scripts/predictor_app.py
```
- Simple prediction interface
- Model selection (Original vs Balanced)
- JSON results output

### 3. Using the App

1. **Open your browser** to `http://127.0.0.1:7870` (or the displayed URL)
2. **Record your voice**: Say "ahhh" clearly for 2-3 seconds
3. **Choose AI model**: Balanced model recommended (92.5% accuracy)
4. **Click "Analyze"**: Get instant results with visualizations
5. **Review results**: Health assessment, confidence scores, and recommendations

## 🤖 AI Models & Performance

### 🎯 **Balanced Model** (Recommended)
- **Accuracy**: 92.5%
- **Technique**: SMOTE (Synthetic Minority Oversampling)
- **Strengths**: Excellent balance between sensitivity and specificity
- **Best for**: General health screening and regular monitoring

### 📊 **Original Model** 
- **Accuracy**: 81.4%
- **Technique**: Standard Random Forest on original data
- **Strengths**: High sensitivity for detecting concerning patterns
- **Best for**: Conservative screening where false negatives are costly

### 🔬 **Technical Specifications**
- **Algorithm**: Random Forest Classifier
- **Features**: 88 voice biomarkers via OpenSMILE eGeMAPSv02
- **Training Data**: Coswara COVID-19 voice dataset
- **Feature Categories**: 
  - 🎵 Pitch & Frequency characteristics
  - 🔊 Loudness & Intensity patterns
  - 🌊 Spectral features (MFCC)
  - 📈 Voice quality measures (jitter, shimmer)
  - 📊 Spectral slopes and bandwidth

## 📊 Visualizations & Analysis

### 🎨 **Voice Analysis Dashboard** (4-Panel Layout)
1. **🌊 Voice Wave Pattern**: Amplitude changes over time with quality statistics
2. **🌈 Voice Frequency Map**: Spectrogram showing pitch distribution with color-coded intensity
3. **📊 Voice Quality Report**: Real-time metrics (steadiness, clarity, pitch, recording quality)
4. **📈 Recording Summary**: Overall quality assessment with readiness indicator

### 🧠 **Explainable AI (SHAP Analysis)**
- **Feature Impact Plots**: Shows how each voice characteristic influences the prediction
- **Force Plots**: Visualizes positive/negative contributions to the final result
- **Feature Importance Rankings**: Top 10 most influential voice biomarkers
- **Personalized Explanations**: Specific to each individual's voice sample

### 📈 **Results Visualization**
- **Confidence Gauges**: AI certainty levels with color-coded indicators
- **Probability Breakdowns**: Detailed percentage scoring for each health category
- **Professional Charts**: Publication-ready matplotlib visualizations

## 🔧 Advanced Usage

### 🏗️ **Building Your Own Dataset**
```bash
# Extract features from Coswara dataset (if available)
python scripts/build_dataset.py

# This creates data/features/voice_dataset.csv with 88 features per sample
```

### 🎓 **Training Custom Models**
```bash
# Train original model
python scripts/train_model.py

# Train balanced model with SMOTE enhancement
python scripts/train_model_balanced.py

# Compare model performances
python scripts/compare_models.py
```

### 🧪 **Testing & Validation**
```bash
# Test individual models
python scripts/test_predictor.py
python scripts/test_balanced_model.py

# Test the complete app interface
python scripts/test_enhanced_app.py
```

## 📋 Dependencies & Requirements

### Core Dependencies
```
gradio>=4.0.0          # Web interface framework
opensmile>=2.4.2       # Voice feature extraction
scikit-learn>=1.3.0    # Machine learning algorithms
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
librosa>=0.10.0        # Audio processing
matplotlib>=3.7.0      # Plotting and visualization
seaborn>=0.12.0        # Statistical visualization
shap>=0.42.0           # Explainable AI
imbalanced-learn>=0.11.0 # SMOTE for balanced training
joblib>=1.3.0          # Model serialization
```

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space (excluding Coswara dataset)
- **CPU**: Multi-core processor recommended for faster training
- **Audio**: Microphone access for voice recording

## 🎯 Use Cases & Applications

### 🏥 **Healthcare Screening**
- **Pre-screening**: Quick voice-based health assessment
- **Remote Monitoring**: Regular health check-ins for chronic conditions
- **Telemedicine**: Voice analysis as part of virtual consultations
- **Public Health**: Large-scale population screening programs

### 🔬 **Research Applications**
- **Voice Biomarker Discovery**: Identify new vocal health indicators
- **Longitudinal Studies**: Track voice changes over time
- **Model Development**: Build specialized models for specific conditions
- **Feature Analysis**: Understand which voice characteristics matter most

### 👥 **Personal Use**
- **Health Monitoring**: Regular self-assessment and trend tracking
- **Early Detection**: Identify potential health changes early
- **Peace of Mind**: Quick reassurance about voice-related health concerns
- **Wellness Tracking**: Include voice health in overall wellness routines

## ⚠️ Important Medical Disclaimer

**VoiceVitals is a research and screening tool, not a medical diagnostic device:**

- ✅ **Use for**: General health awareness, early screening, research purposes
- ❌ **Do not use for**: Medical diagnosis, treatment decisions, emergency situations
- 🩺 **Always consult**: Healthcare professionals for medical concerns
- 📊 **Accuracy**: While our models achieve 92.5% accuracy, individual results may vary
- 🔒 **Privacy**: All voice analysis is performed locally on your device

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 **Bug Reports**
- Use GitHub Issues to report bugs
- Include system information and error messages
- Provide steps to reproduce the issue

### ✨ **Feature Requests**
- Suggest new features or improvements
- Explain the use case and potential impact
- Consider contributing code if possible

### 🔧 **Code Contributions**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📊 **Data Contributions**
- Help improve model accuracy with diverse voice samples
- Contribute to feature extraction techniques
- Share insights from your research applications

## 📚 Technical Documentation

### 🧠 **Machine Learning Pipeline**
1. **Data Collection**: Voice recordings from Coswara dataset
2. **Feature Extraction**: OpenSMILE eGeMAPSv02 (88 features)
3. **Data Preprocessing**: Scaling, balancing with SMOTE
4. **Model Training**: Random Forest with hyperparameter tuning
5. **Validation**: Cross-validation and holdout testing
6. **Deployment**: Gradio web interface with real-time inference

### 🎵 **Voice Feature Categories**
- **Prosodic Features**: Pitch, rhythm, stress patterns
- **Spectral Features**: MFCC, spectral centroids, rolloff
- **Voice Quality**: Jitter, shimmer, harmonics-to-noise ratio
- **Temporal Features**: Speaking rate, pause patterns
- **Energy Features**: Loudness, intensity variations

### 🔮 **Model Architecture**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

## 🎖️ Acknowledgments

- **OpenSMILE Team**: For the excellent feature extraction toolkit
- **Coswara Project**: For providing the COVID voice dataset
- **SHAP Developers**: For making AI explainable and trustworthy
- **Gradio Team**: For the intuitive web interface framework
- **Research Community**: For advancing voice-based health analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact & Support

- **GitHub Issues**: For bug reports and feature requests
- **Email**: [Your contact email]
- **Documentation**: Check the `/docs` folder for detailed guides
- **Community**: Join our discussions in GitHub Discussions

---

**⭐ If VoiceVitals helps your research or projects, please give us a star on GitHub!**

*Built with ❤️ for advancing voice-based health diagnostics and making AI explainable for everyone.*
