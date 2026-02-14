import streamlit as st
import numpy as np
import joblib

# ----------------------
# Load model and scaler
# ----------------------
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")  # scaler fitted on training data WITHOUT id

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
# Prepare input for scaler (exclude id)
# ----------------------
input_features = np.array([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalch, exang, oldpeak, slope, ca, thal
]])

# ----------------------
# Scale input
# ----------------------
input_scaled = scaler.transform(input_features)

# ----------------------
# Add dummy id if model expects it
# ----------------------
id_value = 0
input_final = np.hstack([[id_value], input_scaled])  # final array for model prediction

# ----------------------
# Predict
# ----------------------
if st.button("Predict"):
    if input_final.shape[1] != model.n_features_in_:
        st.error(f"⚠️ Feature count mismatch! "
                 f"Model expects {model.n_features_in_}, input has {input_final.shape[1]}.")
    else:
        prediction = model.predict(input_final)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        # Probability bar if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_final)[0][1]
            st.progress(int(proba*100))
            st.write(f"Risk Probability: {proba*100:.2f}%")

# ----------------------
# Optional: preset buttons for High Risk / Low Risk
# ----------------------
if st.button("Set High Risk Defaults"):
    st.experimental_rerun()  # optional: you can prefill sliders with high-risk values manually

if st.button("Set Low Risk Defaults"):
    st.experimental_rerun()  # optional: prefill sliders with low-risk values manually


