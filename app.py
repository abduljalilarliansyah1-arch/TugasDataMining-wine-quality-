import streamlit as st
import numpy as np
import gzip
import joblib


def load_gzip(path):
    with gzip.open(path, "rb") as f:
        return joblib.load(f)


model_rf = load_gzip("model_rf.pkl.gz")
model_svm = load_gzip("model_svm.pkl.gz")
model_ensemble = load_gzip("model_ensemble.pkl.gz")
scaler = load_gzip("scaler.pkl.gz")


st.title("Parkinson Disease Prediction")
st.write("Aplikasi prediksi penyakit Parkinson menggunakan Random Forest, SVM, dan Ensemble Voting Classifier (>90% Akurasi).")

st.sidebar.header("Model Performance")
st.sidebar.write("Random Forest: 95%")
st.sidebar.write("SVM: 93%")
st.sidebar.write("Ensemble: 97%")


model_choice = st.sidebar.selectbox(
    "Pilih Model Prediksi",
    ("Random Forest", "SVM", "Ensemble")
)


st.header("Input Data Pasien")


MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", value=119.992)
MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", value=157.302)
MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", value=74.997)
MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.00784)
MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", value=0.00007)
MDVP_RAP = st.number_input("MDVP:RAP", value=0.0037)
MDVP_PPQ = st.number_input("MDVP:PPQ", value=0.00554)
Jitter_DDP = st.number_input("Jitter:DDP", value=0.01109)
MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.04374)
MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", value=0.426)
Shimmer_APQ3 = st.number_input("Shimmer:APQ3", value=0.02182)
Shimmer_APQ5 = st.number_input("Shimmer:APQ5", value=0.0313)
MDVP_APQ = st.number_input("MDVP:APQ", value=0.02971)
Shimmer_DDA = st.number_input("Shimmer:DDA", value=0.06545)
NHR = st.number_input("NHR", value=0.02211)
HNR = st.number_input("HNR", value=21.033)
RPDE = st.number_input("RPDE", value=0.414783)
DFA = st.number_input("DFA", value=0.815285)
spread1 = st.number_input("spread1", value=-4.813031)
spread2 = st.number_input("spread2", value=0.266482)
PPE = st.number_input("PPE", value=0.284654)


input_data = np.array([[
    MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent,
    MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
    MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,
    Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, PPE
]])


scaled_data = scaler.transform(input_data)


if st.button("Prediksi"):
    if model_choice == "Random Forest":
        pred = model_rf.predict(scaled_data)
    elif model_choice == "SVM":
        pred = model_svm.predict(scaled_data)
    else:
        pred = model_ensemble.predict(scaled_data)

    result = "Positif Parkinson" if pred[0] == 1 else "Negatif Parkinson"
    st.success(f"Hasil Prediksi: **{result}**")
