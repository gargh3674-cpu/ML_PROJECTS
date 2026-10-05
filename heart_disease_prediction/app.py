# python -m streamlit run app.py

## ye banae ke baad isse stramlit pe deploy karo per usse pehle aapko apni git hub ki repository banani hogi.

'''

Python
  ↓
NumPy / Pandas
  ↓
ML + scikit-learn
  ↓
Streamlit              ← tum yahan ho
  ↓
Git basics              ← NEXT
  ↓
GitHub
  ↓
Deploy Streamlit app
  ↓
Shareable public link 🔗

'''

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Load trained ML artifacts
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "logistic_heart.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
columns = joblib.load(BASE_DIR / "columns.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.16);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
    }

    .hero p {
        margin: 0.55rem 0 0;
        color: #dbeafe;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0.8rem 0 0.8rem;
        color: #0f172a;
    }

    /* Make Streamlit field labels visible on the light background */
    [data-testid="stWidgetLabel"] p,
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stSlider label {
        color: #0f172a !important;
        font-weight: 650 !important;
        opacity: 1 !important;
    }

    /* Keep input text readable */
    [data-baseweb="input"] input,
    [data-baseweb="select"] * {
        color: #0f172a !important;
    }

    .result-card {
        padding: 1.4rem;
        border-radius: 20px;
        background: white;
        border: 1px solid #dbe4f0;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
        margin-top: 1.2rem;
    }

    .result-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
    }

    .warning-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        margin-top: 1rem;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>Enter the patient's clinical information and let the trained Logistic Regression model generate a prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

# -----------------------------
# Inputs
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=54, step=1)
    sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"],
        format_func=lambda x: {
            "ATA": "ATA — Atypical Angina",
            "NAP": "NAP — Non-Anginal Pain",
            "ASY": "ASY — Asymptomatic",
            "TA": "TA — Typical Angina",
        }[x],
    )
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0.0, max_value=300.0, value=130.0, step=1.0)

with col2:
    cholesterol = st.number_input("Cholesterol", min_value=0.0, max_value=1000.0, value=223.0, step=1.0)
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        format_func=lambda x: "No (≤ 120 mg/dl)" if x == 0 else "Yes (> 120 mg/dl)",
    )
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST-T Wave Abnormality",
            "LVH": "Left Ventricular Hypertrophy",
        }[x],
    )
    max_hr = st.number_input("Maximum Heart Rate", min_value=0.0, max_value=300.0, value=138.0, step=1.0)

with col3:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["N", "Y"],
        format_func=lambda x: "No" if x == "N" else "Yes",
    )
    oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=20.0, value=0.6, step=0.1)
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up": "Up",
            "Flat": "Flat",
            "Down": "Down",
        }[x],
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
left, center, right = st.columns([1, 1.4, 1])
with center:
    predict = st.button("🔮 Predict Heart Disease", type="primary", use_container_width=True)

if predict:
    # Start with all model columns as zero.
    input_data = pd.DataFrame(0.0, index=[0], columns=columns)

    # Numerical features that were scaled in the original project.
    numerical_for_scaling = pd.DataFrame(
        {
            "Age": [float(age)],
            "RestingBP": [float(resting_bp)],
            "Cholesterol": [float(cholesterol)],
            "MaxHR": [float(max_hr)],
            "Oldpeak": [float(oldpeak)],
        }
    )
    scaled_values = scaler.transform(numerical_for_scaling)
    scaled_df = pd.DataFrame(scaled_values, columns=numerical_for_scaling.columns)

    for feature in scaled_df.columns:
        input_data[feature] = scaled_df.loc[0, feature]

    # FastingBS was not included in the scaler saved by the notebook.
    input_data["FastingBS"] = fasting_bs

    # Re-create the same one-hot encoded columns used during training.
    input_data[f"Sex_{sex}"] = 1.0
    input_data[f"ChestPainType_{chest_pain}"] = 1.0
    input_data[f"RestingECG_{resting_ecg}"] = 1.0
    input_data[f"ExerciseAngina_{exercise_angina}"] = 1.0
    input_data[f"ST_Slope_{st_slope}"] = 1.0

    # Ensure exact feature order expected by the trained model.
    input_data = input_data[columns]

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ Prediction: Heart Disease Detected")
    else:
        st.success("✅ Prediction: No Heart Disease Detected")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="result-label">Model prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-value">{prediction}</div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown('<div class="result-label">Predicted probability of class 1</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-value">{probability * 100:.2f}%</div>',
            unsafe_allow_html=True,
        )

    st.progress(probability)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="warning-box">This is a machine-learning project demonstration, not a medical diagnosis. A real medical decision should be made by a qualified healthcare professional.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer-note">Built with Python • Streamlit • scikit-learn • Logistic Regression</div>',
    unsafe_allow_html=True,
)
