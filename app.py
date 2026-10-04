"""
Speech Emotion Recognition System - Streamlit Web Application
Analyzes speech audio and predicts emotion (Happy, Sad, Angry, Neutral)
Powered by Librosa, Scikit-Learn, and RAVDESS Speech Dataset.
"""

import os
import io
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa

from model_utils import (
    load_model_pipeline,
    load_audio_file,
    predict_emotion,
    EMOTION_LABELS,
    BASE_DIR,
    SAMPLES_DIR
)

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Modern Dark Glassmorphic Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    
    .badge-blue { background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    .prediction-card {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    
    .metric-title {
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .metric-value {
        color: #F8FAFC;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .metric-unit {
        color: #64748B;
        font-size: 0.75rem;
        margin-left: 2px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    """Load model pipeline once with caching"""
    return load_model_pipeline()


def create_waveform_plot(y, sr, emotion_color='#60A5FA'):
    """Generate high-resolution interactive waveform plot"""
    # Downsample for snappy browser rendering if audio is long
    max_pts = 4000
    if len(y) > max_pts:
        step = len(y) // max_pts
        y_plot = y[::step]
        time_axis = np.linspace(0, len(y) / sr, len(y_plot))
    else:
        y_plot = y
        time_axis = np.linspace(0, len(y) / sr, len(y))
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=y_plot,
        mode='lines',
        line=dict(color=emotion_color, width=1.5),
        fill='tozeroy',
        fillcolor=emotion_color.replace(')', ', 0.15)').replace('rgb', 'rgba') if 'rgb' in emotion_color else 'rgba(96, 165, 250, 0.15)',
        name='Amplitude',
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Audio Waveform (Time Domain)", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title="Time (seconds)", showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(title="Normalized Amplitude", showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color="#94A3B8")
    )
    return fig


def create_spectrogram_plot(y, sr):
    """Generate interactive Mel Spectrogram heatmap"""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    times = librosa.times_like(mel_db, sr=sr)
    freqs = librosa.mel_frequencies(n_mels=64, fmax=8000)
    
    fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title="dB", len=0.9),
        hovertemplate='Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Power: %{z:.1f} dB<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Mel-Frequency Spectrogram (Frequency & Energy)", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title="Time (seconds)", showgrid=False),
        yaxis=dict(title="Frequency (Hz)", showgrid=False),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color="#94A3B8")
    )
    return fig


def create_mfcc_plot(mfcc, sr):
    """Generate interactive MFCC coefficients heatmap"""
    times = librosa.times_like(mfcc, sr=sr)
    fig = go.Figure(data=go.Heatmap(
        z=mfcc,
        x=times,
        y=list(range(1, mfcc.shape[0] + 1)),
        colorscale='Magma',
        colorbar=dict(title="Value", len=0.9),
        hovertemplate='Time: %{x:.2f}s<br>MFCC Coeff: #%{y}<br>Value: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Mel-Frequency Cepstral Coefficients (MFCC 1-40)", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title="Time (seconds)", showgrid=False),
        yaxis=dict(title="MFCC Index", autorange='reversed', showgrid=False),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color="#94A3B8")
    )
    return fig


def create_probability_bar_chart(prob_dict, top_emotion):
    """Create styled horizontal bar chart for emotion probabilities"""
    emotions = list(prob_dict.keys())
    probs = [prob_dict[e] * 100 for e in emotions]
    
    colors = [EMOTION_LABELS[e]['color'] for e in emotions]
    names = [f"{EMOTION_LABELS[e]['emoji']} {EMOTION_LABELS[e]['name']}" for e in emotions]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=probs,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        text=[f"<b>{p:.1f}%</b>" for p in probs],
        textposition='outside',
        textfont=dict(color='#F8FAFC', size=13),
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Emotion Probability Distribution", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title="Probability (%)", range=[0, 115], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13, color="#F8FAFC")),
        margin=dict(l=20, r=30, t=40, b=30),
        height=240,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color="#94A3B8")
    )
    return fig


def create_chroma_plot(chroma_mean):
    """Create radar or bar chart of 12 Pitch Classes (Chroma STFT)"""
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    fig = go.Figure(data=go.Bar(
        x=pitch_names,
        y=chroma_mean,
        marker=dict(
            color=chroma_mean,
            colorscale='Blues',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        hovertemplate='Pitch Class %{x}: %{y:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="12 Chroma Pitch Classes (STFT)", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title="Musical Pitch Class", showgrid=False),
        yaxis=dict(title="Normalized Energy", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=30, r=20, t=40, b=30),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color="#94A3B8")
    )
    return fig


def create_spectral_contrast_plot(contrast_mean):
    """Bar chart for Spectral Contrast octave subbands"""
    bands = [f"Band {i+1}" for i in range(len(contrast_mean))]
    fig = go.Figure(data=go.Bar(
        x=bands,
        y=contrast_mean,
        marker=dict(
            color=contrast_mean,
            colorscale='Plasma',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        hovertemplate='%{x} Contrast: %{y:.2f} dB<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text="Spectral Contrast (Octave Subbands)", font=dict(size=14, color="#E2E8F0")),
        xaxis=dict(title="Frequency Subband", showgrid=False),
        yaxis=dict(title="Contrast (dB)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=30, r=20, t=40, b=30),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color="#94A3B8")
    )
    return fig


def main():
    # Load model pipeline
    model, scaler, encoder, metadata = get_cached_pipeline()
    
    # Header
    st.markdown('<div class="main-title">🎙️ Speech Emotion Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict human emotion from speech acoustics using <b>Librosa</b> and <b>Scikit-Learn</b> Machine Learning.</div>', unsafe_allow_html=True)
    
    # Badges
    st.markdown("""
    <div>
        <span class="badge-pill badge-blue">⚡ Librosa Feature Extraction</span>
        <span class="badge-pill badge-purple">🧠 Ensemble Classifier (MLP + SVC + RF)</span>
        <span class="badge-pill badge-green">📁 RAVDESS Labeled Speech Dataset</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Audio Input & Settings")
        
        input_mode = st.radio(
            "Select Audio Source:",
            ["📁 Upload WAV File", "🎵 Try Sample Audio", "🎙️ Record Live Speech"],
            index=0
        )
        
        st.markdown("---")
        
        st.subheader("🎯 Emotion Reference")
        for emo_key, meta in EMOTION_LABELS.items():
            st.markdown(f"""
            <div style="padding: 8px 12px; margin-bottom: 6px; border-radius: 8px; background: {meta['badge_bg']}; border-left: 4px solid {meta['color']};">
                <span style="font-size: 1.1rem;">{meta['emoji']}</span>
                <strong style="color: {meta['color']}; margin-left: 6px;">{meta['name']}</strong>
                <div style="font-size: 0.75rem; color: #CBD5E1; margin-top: 3px;">{meta['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        st.subheader("📊 System Status")
        if model is not None:
            acc_val = metadata.get('test_accuracy', 0.8439) * 100
            st.success(f"✅ Model Loaded ({acc_val:.1f}% Accuracy)")
            st.caption(f"**Dataset:** RAVDESS ({metadata.get('total_samples', 864)} clips)")
            st.caption(f"**Features:** {metadata.get('feature_dimension', 396)} dimensions")
        else:
            st.error("⚠️ Model artifacts missing!")
            if st.button("🚀 Train Model Now"):
                with st.spinner("Training model on RAVDESS dataset..."):
                    import subprocess
                    subprocess.run(["py", "train_model.py"], check=True)
                    st.cache_resource.clear()
                    st.rerun()

    # Check if model is available
    if model is None:
        st.error("⚠️ The Speech Emotion Recognition model has not been trained yet.")
        st.info("Please click **'Train Model Now'** in the sidebar or run `py train_model.py` in your terminal.")
        return

    # Handle Audio Input Selection
    audio_source_bytes = None
    audio_filename = None
    
    if input_mode == "📁 Upload WAV File":
        uploaded_file = st.file_uploader(
            "Upload speech audio file (.wav format only):",
            type=["wav"],
            help="Upload a mono or stereo WAV recording containing clear spoken words."
        )
        if uploaded_file is not None:
            audio_source_bytes = uploaded_file
            audio_filename = uploaded_file.name
            
    elif input_mode == "🎵 Try Sample Audio":
        st.markdown("Select a preloaded audio clip from the benchmark RAVDESS speech dataset:")
        sample_choice = st.selectbox(
            "Choose a sample emotion clip:",
            [
                ("happy", "😊 Happy Speech Sample (Actor 01 - Joyful tone)"),
                ("sad", "😢 Sad Speech Sample (Actor 01 - Sorrowful tone)"),
                ("angry", "😠 Angry Speech Sample (Actor 01 - Furious tone)"),
                ("neutral", "😐 Neutral Speech Sample (Actor 01 - Calm baseline)")
            ],
            format_func=lambda x: x[1]
        )
        
        sample_path = os.path.join(SAMPLES_DIR, f"sample_{sample_choice[0]}.wav")
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                audio_source_bytes = f.read()
            audio_filename = f"sample_{sample_choice[0]}.wav"
        else:
            st.warning("Sample files not found on disk. Please run `py train_model.py` to generate them.")
            
    elif input_mode == "🎙️ Record Live Speech":
        st.markdown("Speak clearly into your microphone (say any sentence with emotion):")
        mic_audio = st.audio_input("Record your voice")
        if mic_audio is not None:
            audio_source_bytes = mic_audio
            audio_filename = "recorded_speech.wav"

    # Main Processing Section
    if audio_source_bytes is not None:
        try:
            # 1. Load and validate audio
            with st.spinner("Loading audio waveform and validating signal..."):
                y, sr, duration = load_audio_file(audio_source_bytes)
                
            # 2. Audio Player & Info Card
            col_player, col_meta = st.columns([2, 1])
            with col_player:
                st.subheader("🎧 Audio Playback")
                st.audio(audio_source_bytes, format="audio/wav")
                
            with col_meta:
                st.subheader("📋 File Details")
                st.markdown(f"""
                - **File:** `{audio_filename}`
                - **Duration:** `{duration:.2f} seconds`
                - **Sample Rate:** `{sr:,} Hz`
                - **Total Samples:** `{len(y):,}`
                """)

            # 3. Predict Emotion
            with st.spinner("Extracting 396 acoustic features and classifying emotion..."):
                t_start = time.time()
                result = predict_emotion(y, sr, model, scaler, encoder)
                t_infer = (time.time() - t_start) * 1000

            top_emo = result['emotion']
            confidence = result['confidence']
            probabilities = result['probabilities']
            stats = result['stats']
            feature_dict = result['feature_dict']
            emo_meta = EMOTION_LABELS[top_emo]

            st.markdown("---")

            # 4. Prominent Emotion Prediction Display Card
            col_pred_card, col_prob_chart = st.columns([1.1, 1.3])
            
            with col_pred_card:
                st.markdown(f"""
                <div class="prediction-card" style="background: {emo_meta['gradient']}; border-color: {emo_meta['border_color']};">
                    <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: {emo_meta['color']}; letter-spacing: 0.08em; margin-bottom: 8px;">
                        Predicted Emotion
                    </div>
                    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">
                        <span style="font-size: 3.5rem; line-height: 1;">{emo_meta['emoji']}</span>
                        <div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">
                                {emo_meta['name']}
                            </div>
                            <div style="font-size: 1.1rem; font-weight: 600; color: {emo_meta['color']}; margin-top: 4px;">
                                {confidence * 100:.1f}% Confidence
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.88rem; color: #E2E8F0; line-height: 1.45; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;">
                        {emo_meta['description']}
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 8px;">
                        ⚡ Inference time: <b>{t_infer:.1f} ms</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_prob_chart:
                prob_fig = create_probability_bar_chart(probabilities, top_emo)
                st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': False})

            # 5. Acoustic Metric Badges / Tiles
            st.subheader("📊 Extracted Acoustic Metrics")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Duration</div>
                    <div class="metric-value">{stats['duration_s']:.2f}<span class="metric-unit">s</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sample Rate</div>
                    <div class="metric-value">{stats['sample_rate_hz']//1000}<span class="metric-unit">kHz</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">RMS Energy</div>
                    <div class="metric-value">{stats['rms_db']:.1f}<span class="metric-unit">dB</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Zero Crossing</div>
                    <div class="metric-value">{stats['zcr']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m5:
                pitch_display = f"{stats['estimated_pitch_hz']:.0f}" if stats['estimated_pitch_hz'] > 0 else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Pitch (F0)</div>
                    <div class="metric-value">{pitch_display}<span class="metric-unit">Hz</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with m6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Spectral Centroid</div>
                    <div class="metric-value">{stats['spectral_centroid_hz']:.0f}<span class="metric-unit">Hz</span></div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 6. Detailed Interactive Visualizations
            st.subheader("🔍 Audio & Feature Visualizations")
            tab_wave, tab_spec, tab_mfcc, tab_features, tab_model = st.tabs([
                "🌊 Waveform",
                "🌈 Mel Spectrogram",
                "📊 MFCC Heatmap",
                "🔬 Pitch & Spectral Breakdown",
                "🧠 Model & Dataset Architecture"
            ])

            with tab_wave:
                wave_fig = create_waveform_plot(y, sr, emo_meta['color'])
                st.plotly_chart(wave_fig, use_container_width=True)
                st.caption("The time-domain waveform visualizes acoustic amplitude variations, speaking rhythm, and vocal bursts.")

            with tab_spec:
                spec_fig = create_spectrogram_plot(y, sr)
                st.plotly_chart(spec_fig, use_container_width=True)
                st.caption("The Mel Spectrogram maps energy concentration across perceptually spaced frequency bands over time.")

            with tab_mfcc:
                mfcc_fig = create_mfcc_plot(feature_dict['mfcc'], sr)
                st.plotly_chart(mfcc_fig, use_container_width=True)
                st.caption("Mel-Frequency Cepstral Coefficients (MFCCs 1-40) capture vocal tract shape, timbre, and formant envelopes.")

            with tab_features:
                c1, c2 = st.columns(2)
                with c1:
                    chroma_fig = create_chroma_plot(feature_dict['chroma_mean'])
                    st.plotly_chart(chroma_fig, use_container_width=True)
                    st.caption("Chroma features project speech harmony onto the 12 semitone pitch classes.")
                with c2:
                    contrast_fig = create_spectral_contrast_plot(feature_dict['contrast_mean'])
                    st.plotly_chart(contrast_fig, use_container_width=True)
                    st.caption("Spectral Contrast reflects relative spectral peak vs. valley dynamics across 7 octave bands.")

            with tab_model:
                st.markdown("#### 🧠 Architecture & Dataset Transparency")
                
                col_arch1, col_arch2 = st.columns(2)
                with col_arch1:
                    st.markdown("""
                    **Dataset Specifications:**
                    - **Name:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
                    - **Total Speech Files:** 864 audio recordings
                    - **Speakers:** 24 Professional Actors (12 Male, 12 Female)
                    - **Validated Labels:** Neutral (288), Happy (192), Sad (192), Angry (192)
                    - **Sampling Rate:** Standardized at 22,050 Hz with noise reduction & silence trimming
                    
                    **Feature Extraction Pipeline (396 Dimensions):**
                    - **MFCCs:** 40 coefficients (Mean & Std = 80 features)
                    - **Chroma STFT:** 12 pitch classes (Mean & Std = 24 features)
                    - **Mel Spectrogram:** 128 frequency bands (Mean & Std = 256 features)
                    - **Spectral Contrast:** 7 octave bands (Mean & Std = 14 features)
                    - **Tonnetz:** 6 tonal centroid dimensions (Mean & Std = 12 features)
                    - **Time & Spectral Descriptors:** ZCR, RMS Energy, Spectral Centroid, Rolloff, Flatness (10 features)
                    """)
                    
                with col_arch2:
                    st.markdown("""
                    **Ensemble Machine Learning Classifier:**
                    - **Primary Neural Network:** Multi-Layer Perceptron (`MLPClassifier` with 256-128-64 hidden layers, ReLU activation, Adam optimizer, adaptive learning rate)
                    - **Support Vector Classifier:** `SVC` (Radial Basis Function `RBF` kernel, C=5.0)
                    - **Random Forest:** `RandomForestClassifier` (150 trees, max depth 16)
                    - **Ensemble Strategy:** Soft Voting Classifier with weighted posterior confidence
                    - **Test Accuracy:** **84.39%** on unseen hold-out speech samples
                    """)
                    
                st.markdown("#### 📈 Confusion Matrix (Test Set)")
                cm = metadata.get('confusion_matrix', [[35, 2, 0, 1], [5, 28, 2, 3], [0, 0, 54, 4], [1, 2, 7, 29]])
                classes = metadata.get('classes', ['angry', 'happy', 'neutral', 'sad'])
                
                cm_fig = px.imshow(
                    cm,
                    x=[c.capitalize() for c in classes],
                    y=[c.capitalize() for c in classes],
                    labels=dict(x="Predicted Emotion", y="True Emotion", color="Sample Count"),
                    color_continuous_scale="Viridis",
                    text_auto=True
                )
                cm_fig.update_layout(
                    height=280,
                    margin=dict(l=40, r=40, t=30, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    font=dict(color="#E2E8F0")
                )
                st.plotly_chart(cm_fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error analyzing audio file: {str(e)}")
            st.info("💡 **Tip:** Ensure the uploaded file is an uncorrupted WAV audio file containing clear speech (at least 0.5s duration).")
    else:
        # Empty State / Landing Guide
        st.info("👈 Please upload a `.wav` file, select a demo sample clip, or record speech using the sidebar to begin emotion analysis.")
        
        st.markdown("### 🌟 How It Works")
        c_step1, c_step2, c_step3 = st.columns(3)
        with c_step1:
            st.markdown("""
            #### 1. Audio Ingestion
            Upload or record a speech audio clip in WAV format. The audio is loaded and normalized at 22,050 Hz.
            """)
        with c_step2:
            st.markdown("""
            #### 2. Acoustic Feature Extraction
            Librosa extracts 396 multi-dimensional features: MFCCs, Chroma pitch classes, Mel energy, Spectral Contrast, and Tonnetz.
            """)
        with c_step3:
            st.markdown("""
            #### 3. Emotion Classification
            Our calibrated Scikit-Learn ensemble model predicts whether the emotion is **Happy**, **Sad**, **Angry**, or **Neutral** with confidence scores.
            """)


if __name__ == "__main__":
    main()
