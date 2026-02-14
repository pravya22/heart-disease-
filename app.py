import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("heart_model_pipeline.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, step=0.1)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0–3)", [0, 1, 2, 3])

# Create DataFrame (CRITICAL)
input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# Prediction
if st.button("Predict"):
    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.65:
        st.error(f"⚠️ High Risk of Heart Disease ({probability*100:.1f}%)")
    elif probability >= 0.4:
        st.warning(f"⚠️ Moderate Risk ({probability*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk ({probability*100:.1f}%)")
