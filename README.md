# 🎙️ Speech Emotion Recognition System (SER)

An AI-powered Speech Emotion Recognition web application built with **Python**, **Streamlit**, **Librosa**, and **Scikit-Learn**. The system analyzes acoustic properties from spoken speech audio in `.wav` format and accurately classifies the emotional state as **Happy**, **Sad**, **Angry**, or **Neutral**.

---

## 🌟 Key Features

- **🎯 4-Class Emotion Classification**: Detects **Happy 😊**, **Sad 😢**, **Angry 😠**, and **Neutral 😐** with real-time confidence scores.
- **📁 Multi-Source Audio Input**:
  - Upload custom `.wav` audio files (drag & drop support).
  - Test instantly with preloaded **RAVDESS Demo Audio Samples**.
  - Live microphone recording using built-in speech capture.
- **🔬 396-Dimensional Acoustic Feature Extraction**:
  - **MFCCs (1–40)**: Captures vocal tract shape, timber, and phonetic formants.
  - **Chroma STFT**: Analyzes 12 musical pitch classes and tonal harmonies.
  - **Mel Spectrogram (128 Bands)**: Analyzes power spectral energy across human perceptual frequencies.
  - **Spectral Contrast**: Measures peak-to-valley energy dynamics across 7 octave subbands.
  - **Tonnetz**: Captures 6-dimensional tonal centroid and harmonic shifts.
  - **Time & Energy Descriptors**: Zero-Crossing Rate (ZCR), Root-Mean-Square (RMS) Energy, Spectral Centroid, Rolloff, and Flatness.
- **🌊 Interactive Visualizations**:
  - Time-domain Audio Waveform with responsive zoom.
  - Mel-Frequency Spectrogram heatmap in decibels (dB).
  - 40-Coefficient MFCC Feature Heatmap over time frames.
  - Chroma Pitch Class distribution and Spectral Contrast charts.
  - Test Set Confusion Matrix and Performance Breakdown.
- **🛡️ Robust Error Handling**: Validates corrupt audio files, non-audio formats, silent recordings, and short clips (<0.5s).
- **🚀 Trained on Labeled Speech Dataset**: Model trained on **864 speech recordings** from the **RAVDESS** dataset across 24 professional actors, achieving **84.39% Test Accuracy**.

---

## 🧠 System Architecture & Dataset

### Dataset: RAVDESS
The model is trained on the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**:
- **Total Samples:** 864 speech audio files across 24 professional actors (12 male, 12 female).
- **Validated Classes:** Neutral (288), Happy (192), Sad (192), Angry (192).
- **Sampling Rate:** Standardized at 22,050 Hz with silence trimming and volume normalization.

### Machine Learning Ensemble Pipeline
- **Neural Network:** Multi-Layer Perceptron (`MLPClassifier` with 256-128-64 architecture, ReLU, Adam optimizer, early stopping).
- **Support Vector Classifier:** `SVC` (Radial Basis Function RBF kernel, C=5.0).
- **Random Forest:** `RandomForestClassifier` (150 trees, max depth 16).
- **Ensemble Decision:** Soft Voting Classifier aggregating calibrated posterior probabilities.
- **Test Set Accuracy:** **84.39%** (Angry F1: 0.89, Happy F1: 0.80, Neutral F1: 0.89, Sad F1: 0.76).

---

## 📂 Project Directory Structure

```
Speech Emotion Recognition/
│
├── app.py                     # Main Streamlit web application
├── model_utils.py             # Feature extraction & prediction engine
├── train_model.py             # Standalone dataset downloader & model trainer
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation & setup guide
│
├── model/                     # Trained model artifacts & metadata
│   ├── emotion_model.joblib   # Trained Scikit-Learn Ensemble classifier
│   ├── scaler.joblib          # StandardScaler feature normalizer
│   ├── label_encoder.joblib   # LabelEncoder class mappings
│   └── model_metadata.json    # Model evaluation metrics & accuracy
│
├── samples/                   # Preloaded sample audio clips for demo
│   ├── sample_happy.wav       # Sample Happy speech recording
│   ├── sample_sad.wav         # Sample Sad speech recording
│   ├── sample_angry.wav       # Sample Angry speech recording
│   └── sample_neutral.wav     # Sample Neutral speech recording
│
└── dataset/                   # Local dataset cache (RAVDESS)
    └── ravdess/
        ├── Actor_01/
        └── ...
```

---

## 💻 Local Installation & Setup (Windows / VS Code)

### 1. Prerequisites
- **Python 3.10+ or 3.12+** installed on Windows.
- **VS Code** with Python extension installed.

### 2. Open Project in VS Code
1. Open VS Code.
2. Select **File > Open Folder...** and choose the `Speech Emotion Recognition` folder (`c:\Users\prade\Speech Emotion Recognition`).
3. Open a new PowerShell terminal (`Ctrl + ~`).

### 3. Install Required Dependencies
Run the following command in the terminal to install all required libraries:

```powershell
py -m pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the local web server:

```powershell
py -m streamlit run app.py
```

The application will automatically launch in your default web browser at:
👉 **`http://localhost:8501`**

---

## 🔁 (Optional) Retraining the Model from Scratch

If you wish to download the latest audio files and retrain the model:

```powershell
py train_model.py
```

This will automatically:
1. Fetch the RAVDESS speech dataset files.
2. Extract 396-dimensional acoustic features with `librosa`.
3. Train the `MLP + SVC + Random Forest` ensemble.
4. Output classification metrics & confusion matrix.
5. Save updated model weights to the `model/` directory.

---

## 🧪 How to Use the Web App

1. **Choose Audio Input**:
   - Select **"📁 Upload WAV File"** to upload any speech `.wav` file.
   - Or select **"🎵 Try Sample Audio"** to test with verified Happy, Sad, Angry, or Neutral clips.
   - Or select **"🎙️ Record Live Speech"** to record your own voice directly.
2. **Listen & Review**: Play the recording and check file metadata (duration, sample rate).
3. **View Prediction**: Check the predicted emotion card, confidence percentage, and probability distribution across all 4 emotions.
4. **Explore Visualizations**:
   - Navigate through the tabs: **Waveform**, **Mel Spectrogram**, **MFCC Heatmap**, **Pitch & Spectral Breakdown**, and **Model Architecture**.

---

## 📦 Dependencies

- `streamlit` - Web UI framework
- `librosa` - Audio & music signal processing
- `scikit-learn` - Machine learning & neural network classification
- `soundfile` - High-performance audio I/O
- `resampy` - Fast audio resampling
- `plotly` - Interactive data visualizations & charts
- `numpy`, `scipy`, `pandas` - Scientific computing & numerical operations
- `joblib` - Model serialization

---

## 📄 License
This project uses the publicly available [RAVDESS Dataset](https://zenodo.org/records/1188976), distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
