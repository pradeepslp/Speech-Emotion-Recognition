"""
Speech Emotion Recognition - Model Training & Feature Extraction Pipeline
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Target Emotions: Happy, Sad, Angry, Neutral
"""

import os
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import librosa
import soundfile as sf
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model_utils import (
    extract_audio_features,
    EMOTION_LABELS,
    BASE_DIR,
    MODEL_DIR,
    SAMPLES_DIR
)

# RAVDESS emotion mapping:
# 01 = neutral, 02 = calm (mapped to neutral baseline), 03 = happy, 04 = sad, 05 = angry
EMOTION_CODE_MAP = {
    '01': 'neutral',
    '02': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "ravdess")


def download_file(url, save_path):
    """Download single audio file from GitHub repository with retry logic"""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 2000:
        return True
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                if len(content) > 2000:
                    with open(save_path, 'wb') as f:
                        f.write(content)
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def process_audio_file(item):
    """Worker function to extract features from a single audio file"""
    url, file_path, emotion = item
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 2000:
        return None
    try:
        y, sr = librosa.load(file_path, sr=22050)
        features, _ = extract_audio_features(y, sr=sr)
        return (features, emotion, file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def run_training_pipeline():
    """Main pipeline to download data, extract features, train ensemble, and export artifacts"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    print("=" * 65)
    print("Speech Emotion Recognition - RAVDESS Dataset Pipeline")
    print("=" * 65)

    # 1. Prepare download list for all 24 actors
    base_url = "https://raw.githubusercontent.com/ZenvilleErasmus/RAVDESS-emotions-speech-audio-only/master/Audio_Speech_Actors_01-24"
    tasks = []

    for actor_id in range(1, 25):
        actor_str = f"Actor_{actor_id:02d}"
        for emotion_code in ['01', '02', '03', '04', '05']:
            intensities = ['01'] if emotion_code == '01' else ['01', '02']
            for intensity in intensities:
                for statement in ['01', '02']:
                    for repetition in ['01', '02']:
                        filename = f"03-01-{emotion_code}-{intensity}-{statement}-{repetition}-{actor_id:02d}.wav"
                        url = f"{base_url}/{actor_str}/{filename}"
                        save_path = os.path.join(DATASET_DIR, actor_str, filename)
                        tasks.append((url, save_path, EMOTION_CODE_MAP[emotion_code]))

    print(f"Dataset target: {len(tasks)} audio files across 24 actors (12 male, 12 female)")
    
    # Download any missing files
    missing_tasks = [t for t in tasks if not (os.path.exists(t[1]) and os.path.getsize(t[1]) > 2000)]
    if missing_tasks:
        print(f"Downloading {len(missing_tasks)} audio files...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_file, t[0], t[1]) for t in missing_tasks]
            for f in as_completed(futures):
                pass
    else:
        print("All audio files are verified locally in dataset/ravdess/")

    # 2. Extract acoustic features using multi-threading
    print("\nExtracting 396-dimensional acoustic features with Librosa...")
    t0 = time.time()
    
    X = []
    y = []
    sample_files = {'happy': [], 'sad': [], 'angry': [], 'neutral': []}
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_audio_file, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                feat, emo, path = res
                X.append(feat)
                y.append(emo)
                sample_files[emo].append(path)

    X = np.array(X)
    y = np.array(y)
    elapsed = time.time() - t0
    print(f"Feature extraction completed in {elapsed:.1f}s. Extracted shape: {X.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for emo, count in zip(unique, counts):
        print(f"  - {emo.capitalize():8s}: {count} samples")

    # 3. Export high-quality sample clips for the Streamlit demo
    print("\nSaving verified sample audio clips for demo...")
    for emo, paths in sample_files.items():
        if paths:
            # Pick from Actor 01 or Actor 02 for crisp speech demonstration
            chosen = paths[0]
            for p in paths:
                if "Actor_01" in p or "Actor_02" in p:
                    chosen = p
                    break
            dest = os.path.join(SAMPLES_DIR, f"sample_{emo}.wav")
            with open(chosen, 'rb') as src, open(dest, 'wb') as dst:
                dst.write(src.read())
            print(f"  Saved sample: samples/sample_{emo}.wav")

    # 4. Train-Test Split and Standardization
    print("\nPreprocessing data & splitting (80% Train, 20% Test)...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train High-Performance Ensemble Model
    print("\nTraining Multi-Layer Perceptron (MLP) & Ensemble Classifiers...")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.002,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=400,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=25
    )

    svc = SVC(
        C=5.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[('mlp', mlp), ('svc', svc), ('rf', rf)],
        voting='soft',
        weights=[3, 2, 1]
    )

    ensemble.fit(X_train_scaled, y_train)

    # 6. Evaluation
    y_pred = ensemble.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    target_names = [label_encoder.classes_[i] for i in sorted(np.unique(y_test))]

    print("\n" + "=" * 65)
    print(f"Model Performance Metrics (Test Accuracy: {acc * 100:.2f}%)")
    print("=" * 65)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred).tolist()
    print("Confusion Matrix:")
    print(np.array(cm))

    # 7. Serialize Models and Metadata
    print("\nSaving model artifacts to disk...")
    joblib.dump(ensemble, os.path.join(MODEL_DIR, "emotion_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    metadata = {
        "dataset_name": "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)",
        "source": "https://zenodo.org/records/1188976",
        "total_samples": int(len(X)),
        "feature_dimension": int(X.shape[1]),
        "classes": label_encoder.classes_.tolist(),
        "class_counts": {str(k): int(v) for k, v in zip(unique, counts)},
        "test_accuracy": float(round(acc, 4)),
        "confusion_matrix": cm,
        "features": [
            "MFCC (40 coefficients, Mean & Std: 80 features)",
            "Chroma STFT (12 Pitch Classes, Mean & Std: 24 features)",
            "Mel Spectrogram (128 Frequency Bins, Mean & Std: 256 features)",
            "Spectral Contrast (7 Octave Bands, Mean & Std: 14 features)",
            "Tonnetz (6 Tonal Centroid Dimensions, Mean & Std: 12 features)",
            "Zero Crossing Rate (Mean & Std: 2 features)",
            "Root Mean Square Energy (Mean & Std: 2 features)",
            "Spectral Centroid (Mean & Std: 2 features)",
            "Spectral Rolloff (Mean & Std: 2 features)",
            "Spectral Flatness (Mean & Std: 2 features)"
        ],
        "ensemble_models": [
            "Multi-Layer Perceptron (256-128-64 Architecture, Adam)",
            "Support Vector Classifier (RBF Kernel, C=5.0)",
            "Random Forest Classifier (150 Estimators)"
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts successfully saved to: {MODEL_DIR}/")
    print("Pipeline finished successfully!")


if __name__ == "__main__":
    run_training_pipeline()
