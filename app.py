import streamlit as st
import numpy as np
import joblib

# ----------------------
# Load model
# ----------------------
model = joblib.load("heart_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below:")

# ----------------------
# User Inputs
# ----------------------
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# ----------------------
# Encode categorical
# ----------------------
sex = 1 if sex == "Male" else 0

# ----------------------
# Prepare input
# ----------------------
# Check how many features the model expects
expected_features = model.n_features_in_

# Start with dummy id
id_value = 0
input_data = [id_value, age, sex, cp, trestbps, chol, fbs,
              restecg, thalch, exang, oldpeak, slope, ca, thal]

# Make it a numpy array
input_data = np.array([input_data])

# ----------------------
# Check feature count
# ----------------------
if input_data.shape[1] != expected_features:
    st.error(f"⚠️ Input features ({input_data.shape[1]}) "
             f"do not match model expectation ({expected_features}).")
else:
    # ----------------------
    # Predict
    # ----------------------
    if st.button("Predict"):
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        # Show probability if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]
            st.write(f"Risk Probability: {proba*100:.2f}%")


