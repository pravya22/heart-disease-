import streamlit as st
import numpy as np
import joblib

# ----------------------
# Load model and scaler
# ----------------------
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")  # the scaler used during training

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Adjust sliders to test Low Risk and High Risk scenarios:")

# ----------------------
# Preset values
# ----------------------
high_risk_values = {
    "age": 65,
    "sex": 1,       # Male
    "cp": 3,
    "trestbps": 180,
    "chol": 300,
    "fbs": 1,
    "restecg": 1,
    "thalch": 100,
    "exang": 1,
    "oldpeak": 4.0,
    "slope": 2,
    "ca": 3,
    "thal": 2
}

low_risk_values = {
    "age": 40,
    "sex": 0,       # Female
    "cp": 0,
    "trestbps": 110,
    "chol": 180,
    "fbs": 0,
    "restecg": 0,
    "thalch": 170,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 1,
    "ca": 0,
    "thal": 1
}

# ----------------------
# Buttons to fill presets
# ----------------------
preset = st.radio("Select Preset", ["Custom", "High Risk", "Low Risk"])

if preset == "High Risk":
    values = high_risk_values
elif preset == "Low Risk":
    values = low_risk_values
else:
    values = {}

# ----------------------
# User Inputs
# ----------------------
age = st.slider("Age", 1, 120, values.get("age", 60))
sex_input = st.radio("Sex", ["Male", "Female"], index=0 if values.get("sex", 1)==1 else 1)
sex = 1 if sex_input=="Male" else 0
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3], index=values.get("cp",0))
trestbps = st.slider("Resting Blood Pressure", 50, 250, values.get("trestbps",120))
chol = st.slider("Cholesterol", 100, 600, values.get("chol",200))
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0,1], index=values.get("fbs",0))
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2], index=values.get("restecg",0))
thalch = st.slider("Maximum Heart Rate Achieved", 60, 250, values.get("thalch",150))
exang = st.radio("Exercise Induced Angina", [0,1], index=values.get("exang",0))
oldpeak = st.slider("Oldpeak", 0.0, 10.0, values.get("oldpeak",0.5), step=0.1)
slope = st.selectbox("Slope (0-2)", [0,1,2], index=values.get("slope",1))
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3], index=values.get("ca",0))
thal = st.selectbox("Thal (0-3)", [0,1,2,3], index=values.get("thal",1))

# ----------------------
# Prepare input for scaler
# ----------------------
input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalch, exang, oldpeak, slope, ca, thal]])

# ----------------------
# Scale input
# ----------------------
try:
    input_scaled = scaler.transform(input_features)
except ValueError as e:
    st.error(f"Scaler feature mismatch! {e}")
    st.stop()

# ----------------------
# Add dummy id if model expects
# ----------------------
id_value = 0
input_final = np.hstack([[id_value], input_scaled])

# ----------------------
# Predict
# ----------------------
if st.button("Predict"):
    if input_final.shape[1] != model.n_features_in_:
        st.error(f"⚠️ Feature count mismatch! "
                 f"Model expects {model.n_features_in_}, input has {input_final.shape[1]}.")
    else:
        prediction = model.predict(input_final)
        if prediction[0]==1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        # Probability bar if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_final)[0][1]
            st.progress(int(proba*100))
            st.write(f"Risk Probability: {proba*100:.2f}%")
