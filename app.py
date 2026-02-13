import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below:")

# User Inputs
age = st.number_input("Age", min_value=1)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalch = st.number_input("Maximum Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (0-3)", [0,1,2,3])

# Convert categorical
sex = 1 if sex == "Male" else 0

# Create input array
input_data = np.array([[age, trestbps, chol, thalch, oldpeak]])

# Scale numeric columns only
input_data_scaled = scaler.transform(input_data)

# Combine scaled + remaining features
final_input = np.array([[ 
    input_data_scaled[0][0],  # age
    sex,
    cp,
    input_data_scaled[0][1],  # trestbps
    input_data_scaled[0][2],  # chol
    fbs,
    restecg,
    input_data_scaled[0][3],  # thalch
    exang,
    input_data_scaled[0][4],  # oldpeak
    slope,
    ca,
    thal
]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(final_input)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
