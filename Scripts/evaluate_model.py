import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, precision_score, 
                             recall_score, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split

# === Create plots directory if it doesn't exist
import os
os.makedirs("plots", exist_ok=True)

# === Load model, scaler and dataset
model = joblib.load("models/voice_rfc_model_balanced.joblib")
scaler = joblib.load("models/voice_scaler_balanced.joblib")
df = pd.read_csv("data/features/voice_dataset.csv")  # Fixed path to lowercase

X = df.drop(["label", "participant"], axis=1)  # Drop both label and participant columns
y = df["label"]

# === Convert labels to numeric
y_numeric = (y == 'positive').astype(int)  # Convert 'positive' to 1, 'healthy' to 0

# === Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, stratify=y_numeric, test_size=0.2, random_state=42)

# === Scale features
X_test_scaled = scaler.transform(X_test)

# === Predict
y_pred_raw = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# === Convert predictions to numeric
y_pred = (y_pred_raw == 'positive').astype(int)  # Convert string predictions to numeric

# === ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig("plots/roc_curve.png")
plt.close()

# === Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Unhealthy'], yticklabels=['Healthy', 'Unhealthy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("plots/confusion_matrix.png")
plt.close()

# === Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("plots/classification_report.csv")

# === Print Summary with Professional Formatting
print("\n" + "="*50)
print("🎯 MODEL PERFORMANCE EVALUATION")
print("="*50)
print(f"📊 Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"🎯 Precision: {precision_score(y_test, y_pred):.3f}")
print(f"📈 Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"⚖️  F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"🔄 AUC Score: {roc_auc:.3f}")
print("="*50)
print("📁 Plots saved to 'plots/' directory")
print("📋 Classification report saved to 'plots/classification_report.csv'")
