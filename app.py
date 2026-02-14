import streamlit as st
import numpy as np
import joblib

# ----------------------
# Load trained model and scaler
# ----------------------
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")  # the scaler used during training

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Adjust sliders to test Low Risk and High Risk scenarios:")

# ----------------------
# User Inputs
# ----------------------
age = st.slider("Age", 1, 120, 60)
sex = st.radio("Sex", ["Male", "Female"], index=0)
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], index=0)
trestbps = st.slider("Resting Blood Pressure", 50, 250, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=0)
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2], index=0)
thalch = st.slider("Maximum Heart Rate Achieved", 60, 250, 150)
exang = st.radio("Exercise Induced Angina", [0, 1], index=0)
oldpeak = st.slider("Oldpeak", 0.0, 10.0, 0.5, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2], index=1)
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], index=0)
thal = st.selectbox("Thal (0-3)", [0, 1, 2, 3], index=1)

# Encode sex
sex = 1 if sex == "Male" else 0

# ----------------------
# Prepare input (14 features)
# ----------------------
id_value = 0
input_data = np.array([[
    id_value, age, sex, cp, trestbps, chol, fbs, restecg,
    thalch, exang, oldpeak, slope, ca, thal
]])

# ----------------------
# Scale input
# ----------------------
input_scaled = scaler.transform(input_data)

# ----------------------
# Predict
# ----------------------
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # Show probability if model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0][1]
        st.progress(int(proba*100))
        st.write(f"Risk Probability: {proba*100:.2f}%")

