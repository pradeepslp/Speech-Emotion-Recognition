"""
Speech Emotion Recognition - Core Utility & Feature Extraction Module
Shared between training pipeline (train_model.py) and Streamlit web app (app.py).
"""

import os
import io
import json
import warnings
import numpy as np
import librosa
import soundfile as sf
import joblib

# Suppress librosa audio warnings for constantq / resampy
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Emotion mappings & UI metadata
EMOTION_LABELS = {
    'happy': {
        'name': 'Happy',
        'emoji': '😊',
        'color': '#F59E0B',          # Amber / Gold
        'badge_bg': 'rgba(245, 158, 11, 0.15)',
        'border_color': '#F59E0B',
        'gradient': 'linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.05) 100%)',
        'description': 'High pitch variability, elevated acoustic energy, bright spectral brightness, and cheerful melodic intonation.'
    },
    'sad': {
        'name': 'Sad',
        'emoji': '😢',
        'color': '#3B82F6',          # Ocean Blue
        'badge_bg': 'rgba(59, 130, 246, 0.15)',
        'border_color': '#3B82F6',
        'gradient': 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(29, 78, 216, 0.05) 100%)',
        'description': 'Lower fundamental pitch, softer amplitude/energy, subdued vocal dynamics, and elongated syllable duration.'
    },
    'angry': {
        'name': 'Angry',
        'emoji': '😠',
        'color': '#EF4444',          # Crimson Red
        'badge_bg': 'rgba(239, 68, 68, 0.15)',
        'border_color': '#EF4444',
        'gradient': 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.05) 100%)',
        'description': 'Sharp energy bursts, high spectral centroid, elevated zero-crossing rate, and intense vocal tract tension.'
    },
    'neutral': {
        'name': 'Neutral',
        'emoji': '😐',
        'color': '#10B981',          # Emerald Green
        'badge_bg': 'rgba(16, 185, 129, 0.15)',
        'border_color': '#10B981',
        'gradient': 'linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.05) 100%)',
        'description': 'Balanced pitch stability, steady energy envelope, moderate speaking tempo, and calm baseline intonation.'
    }
}

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")


def extract_audio_features(y, sr=22050):
    """
    Extract comprehensive acoustic features from audio signal using Librosa.
    Returns:
        features: 1D numpy array with concatenated acoustic metrics (396 dimensions)
        feature_dict: Dictionary with raw feature arrays for visualization & inspection
    """
    if len(y) == 0:
        raise ValueError("Audio waveform is empty.")
    
    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < int(sr * 0.2):
        y_trimmed = y  # Fallback if audio is very short or over-trimmed
        
    stft = np.abs(librosa.stft(y_trimmed))
    
    # 1. MFCC (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    
    # 2. Chroma STFT (Pitch classes)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    chroma_std = np.std(chroma.T, axis=0)
    
    # 3. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128)
    mel_mean = np.mean(mel.T, axis=0)
    mel_std = np.std(mel.T, axis=0)
    
    # 4. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    contrast_std = np.std(contrast.T, axis=0)
    
    # 5. Tonnetz (Tonal Centroid Features)
    try:
        harmonic = librosa.effects.harmonic(y_trimmed)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        tonnetz_std = np.std(tonnetz.T, axis=0)
    except Exception:
        tonnetz = np.zeros((6, mfcc.shape[1]))
        tonnetz_mean = np.zeros(6)
        tonnetz_std = np.zeros(6)
        
    # 6. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y_trimmed)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # 7. Root Mean Square (RMS) Energy
    rms = librosa.feature.rms(y=y_trimmed)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # 8. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
    spec_cent_mean = np.mean(spec_cent)
    spec_cent_std = np.std(spec_cent)
    
    # 9. Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
    spec_rolloff_mean = np.mean(spec_rolloff)
    spec_rolloff_std = np.std(spec_rolloff)
    
    # 10. Spectral Flatness
    spec_flat = librosa.feature.spectral_flatness(y=y_trimmed)
    spec_flat_mean = np.mean(spec_flat)
    spec_flat_std = np.std(spec_flat)
    
    # Aggregate into 396-dimensional feature vector
    features = np.hstack([
        mfcc_mean, mfcc_std,                # 80
        chroma_mean, chroma_std,            # 24
        mel_mean, mel_std,                  # 256
        contrast_mean, contrast_std,        # 14
        tonnetz_mean, tonnetz_std,          # 12
        zcr_mean, zcr_std,                  # 2
        rms_mean, rms_std,                  # 2
        spec_cent_mean, spec_cent_std,      # 2
        spec_rolloff_mean, spec_rolloff_std,# 2
        spec_flat_mean, spec_flat_std       # 2
    ])
    
    feature_dict = {
        'mfcc': mfcc,
        'chroma': chroma,
        'mel': mel,
        'contrast': contrast,
        'tonnetz': tonnetz,
        'zcr': zcr,
        'rms': rms,
        'spec_cent': spec_cent,
        'spec_rolloff': spec_rolloff,
        'spec_flat': spec_flat,
        'mfcc_mean': mfcc_mean,
        'chroma_mean': chroma_mean,
        'rms_mean': rms_mean,
        'zcr_mean': zcr_mean,
        'spec_cent_mean': spec_cent_mean,
        'contrast_mean': contrast_mean,
        'tonnetz_mean': tonnetz_mean
    }
    
    return features, feature_dict


def load_audio_file(file_or_bytes, target_sr=22050):
    """
    Safely load audio file or uploaded file buffer into numpy array and sample rate.
    Includes validation for corrupt, empty, silent, or non-audio inputs.
    """
    if file_or_bytes is None:
        raise ValueError("No audio file was provided.")
        
    try:
        if isinstance(file_or_bytes, (str, os.PathLike)):
            if not os.path.exists(file_or_bytes) or os.path.getsize(file_or_bytes) == 0:
                raise ValueError("The audio file is empty or missing from disk.")
            y, sr = librosa.load(file_or_bytes, sr=target_sr)
        else:
            file_bytes = file_or_bytes.read() if hasattr(file_or_bytes, 'read') else file_or_bytes
            if len(file_bytes) < 44:  # WAV header is at least 44 bytes
                raise ValueError("The uploaded file is too small to be a valid audio file.")
            buffer = io.BytesIO(file_bytes)
            y, sr = librosa.load(buffer, sr=target_sr)
            
        if len(y) == 0:
            raise ValueError("Audio signal is empty or contains no readable data.")
            
        duration = len(y) / sr
        if duration < 0.3:
            raise ValueError(f"Audio duration ({duration:.2f}s) is too short. Please provide at least 0.5 seconds of speech.")
            
        # Check if audio is completely silent
        max_amp = np.max(np.abs(y))
        if max_amp < 1e-4:
            raise ValueError("The audio file contains only silence. Please provide a clip with audible speech.")
            
        return y, sr, duration
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Invalid or corrupted audio file: {str(e)}. Please upload a valid WAV audio file.")


def load_model_pipeline():
    """
    Load serialized model, scaler, and label encoder.
    """
    model_path = os.path.join(MODEL_DIR, "emotion_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        return None, None, None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
    return model, scaler, encoder, metadata


def predict_emotion(y, sr, model, scaler, encoder):
    """
    Predict emotion class and class probabilities for given audio signal.
    """
    features, feature_dict = extract_audio_features(y, sr=sr)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict probabilities and class
    probs = model.predict_proba(features_scaled)[0]
    classes = encoder.classes_
    
    # Format probabilities into dictionary
    prob_dict = {cls_name: float(p) for cls_name, p in zip(classes, probs)}
    
    # Ensure all 4 target emotions are present
    for emo in ['happy', 'sad', 'angry', 'neutral']:
        if emo not in prob_dict:
            prob_dict[emo] = 0.0
            
    top_emotion = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[top_emotion]
    
    # Compute high-level acoustic statistics for display
    rms_db = float(20 * np.log10(max(feature_dict['rms_mean'], 1e-6)))
    zcr_val = float(feature_dict['zcr_mean'])
    spec_cent_hz = float(feature_dict['spec_cent_mean'])
    
    # Pitch estimation
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=500)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]
        avg_pitch = float(np.median(pitch_values)) if len(pitch_values) > 0 else 0.0
    except Exception:
        avg_pitch = 0.0
        
    stats = {
        'duration_s': float(len(y) / sr),
        'sample_rate_hz': int(sr),
        'rms_db': round(rms_db, 2),
        'zcr': round(zcr_val, 4),
        'spectral_centroid_hz': round(spec_cent_hz, 1),
        'estimated_pitch_hz': round(avg_pitch, 1)
    }
    
    return {
        'emotion': top_emotion,
        'confidence': confidence,
        'probabilities': prob_dict,
        'stats': stats,
        'features': features,
        'feature_dict': feature_dict
    }
