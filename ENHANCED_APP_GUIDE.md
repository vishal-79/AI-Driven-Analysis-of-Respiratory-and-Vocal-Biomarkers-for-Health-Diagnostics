# 🎤 VoiceVitals: Enhanced AI Health Classifier

## 🚀 Complete Feature Overview

### ✅ **Successfully Implemented Features:**

#### 1. **🔍 SHAP Explanations**
- **Individual Prediction Analysis**: SHAP values for each voice sample
- **Feature Impact Visualization**: Shows how each feature influenced the current prediction
- **Explainable AI**: Provides personalized explanations for every prediction

#### 2. **📊 Matplotlib Feature Importance Charts**
- **Top 10 Features**: Beautiful horizontal bar charts showing most important voice biomarkers
- **Color-coded Visualization**: Viridis colormap for enhanced readability
- **Customized Styling**: Professional charts with value labels and grid lines

#### 3. **📋 Complete 88-Feature Table**
- **Detailed Breakdown**: All 88 voice biomarkers with values and SHAP impacts
- **Categorized Features**: Grouped by type (Pitch, Loudness, Spectral, Voice Quality, etc.)
- **Interactive Table**: Sortable and searchable DataFrame in Gradio
- **Impact Direction**: Shows whether each feature pushes toward Healthy or COVID-Positive

#### 4. **🎛️ Tabbed Gradio Layout**
- **4 Main Tabs**: Health Prediction, Feature Importance, SHAP Analysis, Detailed Features
- **Clean Interface**: Organized and professional user experience
- **Model Selection**: Choose between Balanced (92.5%) or Original (81.4%) models
- **Collapsible Info**: Technical details in expandable accordion

## 🎯 **Tab Breakdown:**

### Tab 1: 🏥 Health Prediction
```json
{
  "🏥 Health Status": "✅ Healthy (healthy)",
  "🎯 Confidence Score": "70.0%",
  "📊 Probability Breakdown": "Healthy: 70.0% | COVID-Positive: 30.0%",
  "🤖 Model Information": "🎯 Balanced Model (SMOTE - 92.5% accuracy)",
  "🔬 Features Analyzed": "88 voice biomarkers",
  "📈 Prediction Strength": "Medium Confidence"
}
```

### Tab 2: 📊 Feature Importance
- **Visual Chart**: Top 10 most important features across all predictions
- **Horizontal Bars**: Easy-to-read importance scores
- **Feature Names**: Full technical names of voice biomarkers
- **Color Gradient**: Visual hierarchy of importance

### Tab 3: 🎯 SHAP Analysis
- **Personalized Explanation**: Feature impacts for THIS specific prediction
- **SHAP Summary Plot**: Bar chart showing feature contributions
- **Impact Direction**: Positive/negative influence on prediction
- **Research-Grade**: Uses TreeExplainer for Random Forest models

### Tab 4: 📋 Detailed Features
| Rank | Feature Name | Category | Value | SHAP Impact | Impact Direction |
|------|--------------|----------|-------|-------------|-----------------|
| 1 | slopeUV0-500_sma3nz_amean | 📊 Spectral Slopes | 0.1234 | 0.0567 | → Positive |
| 2 | loudness_sma3_meanFallingSlope | 🔊 Loudness & Intensity | -0.5678 | -0.0234 | → Negative |
| ... | ... | ... | ... | ... | ... |

## 🔧 **Technical Implementation:**

### **Dependencies Installed:**
- ✅ `shap` - For explainable AI analysis
- ✅ `matplotlib` - For professional charts and plots
- ✅ `seaborn` - Enhanced statistical visualizations
- ✅ `gradio` - Web interface framework
- ✅ `opensmile` - Voice feature extraction
- ✅ `scikit-learn` - Machine learning models

### **Model Support:**
- **Balanced Model**: SMOTE-enhanced, 92.5% accuracy, better balanced performance
- **Original Model**: Baseline model, 81.4% accuracy, higher COVID sensitivity

### **Voice Features Analyzed (88 total):**
1. **🎵 Pitch & Frequency** (F0, formants)
2. **🔊 Loudness & Intensity** (energy, dynamics)
3. **🌊 Spectral Features** (MFCCs, spectral shape)
4. **📈 Voice Quality** (jitter, shimmer, HNR)
5. **📊 Spectral Slopes** (frequency distribution)
6. **📐 Bandwidth Features** (formant bandwidths)
7. **🔧 Other Features** (alpha ratio, voiced segments)

## 🎤 **User Experience:**

### **Input Methods:**
- 🎙️ **Direct Recording**: Record audio directly in browser
- 📁 **File Upload**: Upload WAV/MP3 files
- 🎵 **Example Audio**: Pre-loaded sample for testing

### **Output Formats:**
- 📊 **JSON Results**: Structured prediction data
- 🖼️ **PNG Charts**: High-resolution matplotlib plots
- 📋 **DataFrame Tables**: Interactive feature analysis
- 📖 **Markdown Guides**: Helpful explanations

## 🌐 **Access Information:**

- **Local URL**: http://127.0.0.1:7861
- **Port**: 7861 (to avoid conflict with other apps)
- **Sharing**: Set to local access only for security

## 🔬 **Research Applications:**

### **Healthcare Research:**
- Voice biomarker discovery
- Respiratory condition screening
- Longitudinal health monitoring

### **AI Development:**
- Explainable AI in healthcare
- Feature importance analysis
- Model interpretability studies

### **Educational Use:**
- Demonstration of ML in healthcare
- Voice analysis techniques
- Explainable AI concepts

## ⚠️ **Important Notes:**

1. **Research Tool**: For educational and research purposes only
2. **Not Medical Device**: Should not replace professional healthcare
3. **Data Privacy**: All processing done locally
4. **Model Limitations**: Trained on specific dataset conditions

## 🚀 **Quick Start:**

1. **Launch App**: Run `python scripts/predictor_app_complete.py`
2. **Open Browser**: Navigate to http://127.0.0.1:7861
3. **Upload Audio**: Record or upload a clear 'ahhh' sound (2-3 seconds)
4. **Select Model**: Choose Balanced (recommended) or Original
5. **Analyze**: Click "🔍 Analyze Voice" button
6. **Explore Results**: View all 4 tabs for complete analysis

The enhanced VoiceVitals app now provides comprehensive, explainable AI-powered voice health analysis with professional visualizations and detailed feature breakdowns! 🎉
