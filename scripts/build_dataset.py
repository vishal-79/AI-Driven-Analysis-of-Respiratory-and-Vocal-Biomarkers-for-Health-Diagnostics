import os
import pandas as pd
from opensmile import Smile, FeatureSet, FeatureLevel

# === CONFIGURATION ===
BASE_DIR = 'datasets/Coswara'  # Fixed case-sensitive path
AUDIO_TYPE = 'vowel-a.wav'   # Change this to other types like 'cough-heavy.wav' if needed
MAX_SAMPLES = 300            # Limit to avoid long runs while testing
HEALTHY_LABELS = ['healthy']  # Healthy participants
POSITIVE_LABELS = ['positive_mild', 'positive_moderate', 'positive_asymp']  # COVID-positive participants

# === Initialize OpenSMILE extractor ===
smile = Smile(
    feature_set=FeatureSet.eGeMAPSv02,
    feature_level=FeatureLevel.Functionals
)

# === Load Coswara metadata ===
meta_path = os.path.join(BASE_DIR, 'combined_data.csv')
df_meta = pd.read_csv(meta_path)

# === Filter participants based on COVID status
ALL_LABELS = HEALTHY_LABELS + POSITIVE_LABELS
df_meta = df_meta[df_meta['covid_status'].isin(ALL_LABELS)]

print(f"🧪 Found {len(df_meta)} participants with covid_status in {ALL_LABELS}")

features_all = []
count = 0

# === Loop through filtered participants
for idx, row in df_meta.iterrows():
    folder = row['id']  # 'id' is the folder name
    covid_status = row['covid_status']
    
    # Map COVID status to binary label
    if covid_status in HEALTHY_LABELS:
        label = 'healthy'
    elif covid_status in POSITIVE_LABELS:
        label = 'positive'
    else:
        continue  # Skip unknown labels
    
    found = False

    # Search for participant folder inside each date folder
    for date_folder in os.listdir(os.path.join(BASE_DIR, 'Extracted_data')):
        participant_path = os.path.join(BASE_DIR, 'Extracted_data', date_folder, folder)
        audio_path = os.path.join(participant_path, AUDIO_TYPE)

        if os.path.exists(audio_path):
            try:
                # Extract features using OpenSMILE
                features = smile.process_file(audio_path)
                features['label'] = label
                features['participant'] = folder
                features_all.append(features)
                count += 1

                if count % 25 == 0:
                    print(f"✅ Processed {count} samples...")

                if count >= MAX_SAMPLES:
                    break
            except Exception as e:
                print(f"⚠️ Error processing {audio_path}: {e}")
            found = True
            break

    if not found:
        print(f"❌ File not found: {folder}/{AUDIO_TYPE}")

    if count >= MAX_SAMPLES:
        break

# === Save extracted features to CSV
if features_all:
    df_full = pd.concat(features_all)
    df_full.reset_index(drop=True, inplace=True)
    os.makedirs('data/features', exist_ok=True)
    output_path = 'data/features/voice_dataset.csv'
    df_full.to_csv(output_path, index=False)
    print(f"\n📁 Saved dataset with {len(df_full)} rows to: {output_path}")
else:
    print("❌ No features were extracted. Please check the file paths.")
