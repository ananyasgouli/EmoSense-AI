import streamlit as st
import numpy as np
import librosa
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="EmoSense AI",
    page_icon="🎧",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #100d25, #171438, #101b35);
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #29234f;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #d9c5ff !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #f2edff !important;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #5636a8, #176b9b);
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 5px;
}
.hero p {
    color: #e6e6ff;
    font-size: 17px;
}
.metric-card {
    background: #211e43;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #51477e;
    text-align: center;
}
.metric-card h2 {
    color: #c6a7ff;
}
.info-card {
    background: #211e43;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #51477e;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "emotion_svm_model_v2.pkl")

    with open(BASE_DIR / "emotion_labels.json", "r") as f:
        labels = json.load(f)

    return model, labels

try:
    model, emotion_labels = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---------------- FEATURE EXTRACTION ----------------

def extract_features(audio_file):

    audio, sr = librosa.load(
        audio_file,
        sr=16000
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # MFCC delta
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Pitch
    pitch = librosa.yin(
        audio,
        fmin=65,
        fmax=500,
        sr=sr
    )

    pitch_features = [
        np.mean(pitch),
        np.std(pitch)
    ]

    # Energy
    rms = librosa.feature.rms(y=audio)[0]

    energy_features = [
        np.mean(rms),
        np.std(rms)
    ]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    centroid_features = [
        np.mean(centroid),
        np.std(centroid)
    ]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    zcr_features = [
        np.mean(zcr),
        np.std(zcr)
    ]

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        pitch_features,
        energy_features,
        centroid_features,
        zcr_features
    ])

    if len(features) != 128:
        raise ValueError(
            f"Expected 128 features, received {len(features)}"
        )

    return audio, sr, features.reshape(1, -1)

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🎧 EmoSense AI</h1>
    <p>AI-Powered Speech Emotion Recognition</p>
    <p>Discover the emotions expressed through speech</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🎧 EmoSense AI")
    st.markdown("---")
    st.subheader("Supported Emotions")

    for emotion in emotion_labels:
        st.write("•", emotion)

    st.markdown("---")
    st.caption("Speech Emotion Recognition using Machine Learning")
    st.caption("Dataset: RAVDESS Speech")

# ---------------- MAIN INTERFACE ----------------

st.subheader("🎙️ Upload Your Audio")

st.write(
    "Upload a speech recording to explore the model's emotion predictions."
)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("✨ Analyze Speech", use_container_width=True):

        with st.spinner("Analyzing speech features..."):

            try:
                audio, sr, features = extract_features(
                    uploaded_file
                )

                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                predicted_emotion = emotion_labels[prediction]
                confidence = float(np.max(probabilities)) * 100

                st.markdown("---")
                st.subheader("📊 Emotion Analysis Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>Predicted Emotion</h2>
                        <h1>{predicted_emotion}</h1>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>Highest Model Score</h2>
                        <h1>{confidence:.2f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Probability chart
                st.subheader("Emotion Probability Distribution")

                chart_data = {
                    "Emotion": emotion_labels,
                    "Score (%)": probabilities * 100
                }

                fig = px.bar(
                    chart_data,
                    x="Emotion",
                    y="Score (%)",
                    color="Score (%)",
                    color_continuous_scale="Purples",
                    text_auto=".1f"
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#171438",
                    plot_bgcolor="#171438",
                    font_color="white",
                    yaxis_range=[0, 100]
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Top three
                st.subheader("🏆 Top 3 Predictions")

                top_indices = np.argsort(probabilities)[::-1][:3]

                for rank, index in enumerate(top_indices, 1):
                    st.write(
                        f"{rank}. **{emotion_labels[index]}** — "
                        f"{probabilities[index] * 100:.2f}%"
                    )

                # Waveform
                st.markdown("---")
                st.subheader("🎵 Audio Waveform")

                fig_wave = go.Figure()

                fig_wave.add_trace(
                    go.Scatter(
                        y=audio,
                        mode="lines",
                        name="Audio signal"
                    )
                )

                fig_wave.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#171438",
                    plot_bgcolor="#171438",
                    xaxis_title="Audio Samples",
                    yaxis_title="Amplitude"
                )

                st.plotly_chart(
                    fig_wave,
                    use_container_width=True
                )

                st.success("Speech analysis completed!")

                st.info(
                    "This model estimates emotion-related patterns "
                    "in speech. Its predictions are experimental and "
                    "do not establish a person's actual feelings."
                )

            except Exception as e:
                st.error(f"Analysis failed: {e}")

else:
    st.markdown("""
    <div class="info-card">
        <h3>💡 How to use EmoSense AI</h3>
        <p>1. Upload a speech audio file.</p>
        <p>2. Click Analyze Speech.</p>
        <p>3. Explore the predicted emotion and probability chart.</p>
        <p>4. View the audio waveform.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    "<center>EmoSense AI | Speech Emotion Recognition Project</center>",
    unsafe_allow_html=True
)