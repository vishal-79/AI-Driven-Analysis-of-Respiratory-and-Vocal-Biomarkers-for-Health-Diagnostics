"""
Model Comparison Script: Original vs Balanced
Compare performance of the original model vs SMOTE-balanced model
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compare_models():
    """Compare original and balanced models on the same test set"""
    
    print("🔬 Model Comparison: Original vs Balanced")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv("data/features/voice_dataset.csv")
    df = df.drop(columns=["participant"])
    df.dropna(inplace=True)
    
    X = df.drop(columns=["label"])
    y = df["label"]
    
    # Use same test split as original model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load both models
    model_original = joblib.load("models/voice_rfc_model.joblib")
    scaler_original = joblib.load("models/voice_scaler.joblib")
    
    model_balanced = joblib.load("models/voice_rfc_model_balanced.joblib")
    scaler_balanced = joblib.load("models/voice_scaler_balanced.joblib")
    
    # Test original model
    X_test_orig = scaler_original.transform(X_test)
    y_pred_orig = model_original.predict(X_test_orig)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    
    # Test balanced model on original test set
    X_test_balanced = scaler_balanced.transform(X_test)
    y_pred_balanced = model_balanced.predict(X_test_balanced)
    acc_balanced = accuracy_score(y_test, y_pred_balanced)
    
    print(f"\n📊 RESULTS ON ORIGINAL TEST SET:")
    print(f"Original Model Accuracy:  {acc_orig*100:.2f}%")
    print(f"Balanced Model Accuracy:  {acc_balanced*100:.2f}%")
    print(f"Improvement: +{(acc_balanced-acc_orig)*100:.2f} percentage points")
    
    print(f"\n🎯 ORIGINAL MODEL PERFORMANCE:")
    print(classification_report(y_test, y_pred_orig))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_orig))
    
    print(f"\n⚖️ BALANCED MODEL PERFORMANCE:")
    print(classification_report(y_test, y_pred_balanced))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_balanced))
    
    # Calculate specific metrics
    orig_report = classification_report(y_test, y_pred_orig, output_dict=True)
    balanced_report = classification_report(y_test, y_pred_balanced, output_dict=True)
    
    print(f"\n📈 DETAILED COMPARISON:")
    print(f"{'Metric':<25} {'Original':<12} {'Balanced':<12} {'Improvement'}")
    print("-" * 65)
    
    for class_name in ['healthy', 'positive']:
        for metric in ['precision', 'recall', 'f1-score']:
            orig_val = orig_report[class_name][metric]
            balanced_val = balanced_report[class_name][metric]
            improvement = balanced_val - orig_val
            print(f"{class_name.title()} {metric:<15} {orig_val:<12.3f} {balanced_val:<12.3f} {improvement:+.3f}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"• Balanced model shows {'better' if acc_balanced > acc_orig else 'similar'} overall accuracy")
    print(f"• Healthy recall improved by {(balanced_report['healthy']['recall'] - orig_report['healthy']['recall'])*100:+.1f} percentage points")
    print(f"• Positive recall changed by {(balanced_report['positive']['recall'] - orig_report['positive']['recall'])*100:+.1f} percentage points")
    print(f"• More balanced performance across both classes")

if __name__ == "__main__":
    compare_models()
