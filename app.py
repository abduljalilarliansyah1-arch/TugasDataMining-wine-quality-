# app.py - Wine Quality Prediction (RandomForest + SVM + Ensemble)
import streamlit as st
import numpy as np
import gzip
import joblib
from io import BytesIO

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="centered")

# -----------------------------
# Helper: load compressed joblib (.pkl.gz)
# -----------------------------
def load_gzip_joblib(path):
    with gzip.open(path, "rb") as f:
        return joblib.load(f)

# -----------------------------
# Load models and scaler (compressed .pkl.gz expected)
# -----------------------------
@st.cache_resource
def load_models():
    try:
        rf = load_gzip_joblib("model_rf.pkl.gz")
        svm = load_gzip_joblib("model_svm.pkl.gz")
        ensemble = load_gzip_joblib("model_ensemble.pkl.gz")
        scaler = load_gzip_joblib("scaler.pkl.gz")
        return rf, svm, ensemble, scaler
    except Exception as e:
        # return None set so app can show friendly error
        return None, None, None, None

rf, svm, ensemble, scaler = load_models()

# -----------------------------
# Page header
# -----------------------------
st.title("🍷 Wine Quality Classifier")
st.markdown("""
Prediksi kualitas wine: **Good (quality ≥ 7)** vs **Bad (quality < 7)**.  
Model: **Random Forest**, **SVM**, dan **Ensemble (soft voting)**.  
Pastikan file `model_*.pkl.gz` dan `scaler.pkl.gz` ada di folder yang sama dengan `app.py`.
""")

# show small note with uploaded screenshot path if needed
st.caption("Screenshot referensi (uploaded): `/mnt/data/334e8ff0-eeae-4163-8a22-2e6b77ebd3ec.png`")

if any(m is None for m in [rf, svm, ensemble, scaler]):
    st.error("⚠️ Model atau scaler tidak ditemukan. Upload file `.pkl.gz` (model_rf.pkl.gz, model_svm.pkl.gz, model_ensemble.pkl.gz, scaler.pkl.gz) ke folder yang sama.")
    st.stop()

# -----------------------------
# Sidebar: model performance (put your real values here)
# -----------------------------
st.sidebar.header("Model Performance (Test set)")
st.sidebar.write("Random Forest: **~92%**")
st.sidebar.write("SVM: **~93%**")
st.sidebar.write("Ensemble: **~94%**")
st.sidebar.markdown("---")
st.sidebar.write("Tips: pastikan input angka realistis sesuai distribusi dataset (lihat README).")

# -----------------------------
# Feature inputs (11 fitur Wine)
# -----------------------------
st.subheader("Masukkan 11 fitur Wine (angka):")

# Provide sensible defaults from typical wine dataset ranges
fixed_acidity = st.number_input("fixed acidity", min_value=0.0, step=0.01, value=7.0, format="%.2f")
volatile_acidity = st.number_input("volatile acidity", min_value=0.0, step=0.001, value=0.27, format="%.3f")
citric_acid = st.number_input("citric acid", min_value=0.0, step=0.001, value=0.33, format="%.3f")
residual_sugar = st.number_input("residual sugar", min_value=0.0, step=0.01, value=6.0, format="%.2f")
chlorides = st.number_input("chlorides", min_value=0.0, step=0.0001, value=0.045, format="%.4f")
free_sulfur_dioxide = st.number_input("free sulfur dioxide", min_value=0.0, step=0.1, value=30.0, format="%.1f")
total_sulfur_dioxide = st.number_input("total sulfur dioxide", min_value=0.0, step=0.1, value=115.0, format="%.1f")
density = st.number_input("density", min_value=0.0, step=0.000001, value=0.994, format="%.6f")
pH = st.number_input("pH", min_value=0.0, step=0.01, value=3.2, format="%.2f")
sulphates = st.number_input("sulphates", min_value=0.0, step=0.01, value=0.5, format="%.2f")
alcohol = st.number_input("alcohol", min_value=0.0, step=0.1, value=10.5, format="%.1f")

# Collect into numpy array
input_features = np.array([[ 
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
    density, pH, sulphates, alcohol
]])

# -----------------------------
# Model selection and prediction
# -----------------------------
model_choice = st.selectbox("Pilih model:", ["Ensemble", "Random Forest", "SVM"])

if st.button("🔍 Prediksi"):
    try:
        # scale
        scaled = scaler.transform(input_features)  # must match training shape (11 features)
    except Exception as e:
        st.error(f"Scaler error: {e}")
        st.stop()

    if model_choice == "Random Forest":
        pred = rf.predict(scaled)[0]
        proba = rf.predict_proba(scaled)[0]
    elif model_choice == "SVM":
        pred = svm.predict(scaled)[0]
        proba = svm.predict_proba(scaled)[0]
    else:
        pred = ensemble.predict(scaled)[0]
        proba = ensemble.predict_proba(scaled)[0]

    label = "Good (quality ≥ 7)" if pred == 1 else "Bad (quality < 7)"
    color = st.success if pred == 1 else st.error
    color(f"**Hasil Prediksi: {label}**")

    # Show probabilities neatly
    st.write(f"Confidence: Good = **{proba[1]*100:.2f}%**, Bad = **{proba[0]*100:.2f}%**")

    # Optional: show feature values and scaled version
    with st.expander("Detail input & scaled features"):
        st.write("Input fitur (original):")
        st.write(dict(zip([
            "fixed_acidity","volatile_acidity","citric_acid","residual_sugar",
            "chlorides","free_sulfur_dioxide","total_sulfur_dioxide",
            "density","pH","sulphates","alcohol"
        ], input_features.flatten().tolist())))
        st.write("Scaled (what model sees):")
        st.write(scaled.tolist())
