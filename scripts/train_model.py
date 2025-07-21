import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === Load the extracted feature dataset ===
print("📊 Loading voice dataset...")
df = pd.read_csv('data/features/voice_dataset.csv')
print(f"Original dataset shape: {df.shape}")

# === Drop unused columns ===
df = df.drop(columns=['participant'])

# === Drop rows with NaNs (from short/invalid audio segments) ===
print(f"NaN values found: {df.isna().sum().sum()}")
df.dropna(inplace=True)
print(f"Shape after cleaning: {df.shape}")

# === Separate features and labels ===
X = df.drop(columns=['label'])
y = df['label']
print(f"Features: {X.shape}, Labels: {y.shape}")
print(f"Label distribution:\n{y.value_counts()}")

# === Scale the features ===
print("\n🔧 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split into training and testing sets ===
print("📂 Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# === Train Random Forest Classifier ===
print("\n🌲 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate on test set ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧱 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save the model and scaler ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/voice_rfc_model.joblib")
joblib.dump(scaler, "models/voice_scaler.joblib")
print("\n💾 Model and scaler saved to 'models/'")

# === Feature Importance Analysis ===
print("\n🔍 Top 10 Most Important Features:")
feature_names = X.columns
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")

print(f"\n📈 Model Performance Summary:")
print(f"   • Accuracy: {acc*100:.2f}%")
print(f"   • Precision (Healthy): {classification_report(y_test, y_pred, output_dict=True)['healthy']['precision']:.3f}")
print(f"   • Recall (Healthy): {classification_report(y_test, y_pred, output_dict=True)['healthy']['recall']:.3f}")
print(f"   • Precision (Positive): {classification_report(y_test, y_pred, output_dict=True)['positive']['precision']:.3f}")
print(f"   • Recall (Positive): {classification_report(y_test, y_pred, output_dict=True)['positive']['recall']:.3f}")
print(f"   • Dataset: {len(df)} samples ({len(y[y=='healthy'])} healthy, {len(y[y=='positive'])} positive)")
print(f"   • Features: {X.shape[1]} voice biomarkers")
