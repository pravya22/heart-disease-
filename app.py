import streamlit as st
import numpy as np
import joblib

# ----------------------
# Load model and scaler
# ----------------------
model = joblib.load("heart_model.pkl")
heart_model.pkl = joblib.load("heart_model.pkl")

# safe handling if scaler has no feature names
if hasattr(heart_model, "feature_names_in_"):
    heart_model_names = list(heart_model_feature_names_in_)
else:
    heart_model_names = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# ----------------------
# Page setup
# ----------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Use presets or custom values to predict heart disease risk.")

# ----------------------
# Preset values
# ----------------------
high_risk_values = {
    "age": 65, "sex": 1, "cp": 3, "trestbps": 180, "chol": 300,
    "fbs": 1, "restecg": 1, "thalch": 100, "exang": 1,
    "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 2
}

low_risk_values = {
    "age": 40, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
    "fbs": 0, "restecg": 0, "thalch": 170, "exang": 0,
    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 1
}

preset = st.radio("Select Preset", ["Custom", "High Risk", "Low Risk"])

if preset == "High Risk":
    values = high_risk_values
elif preset == "Low Risk":
    values = low_risk_values
else:
    values = {}

# ----------------------
# Helper for selectbox index
# ----------------------
def idx(options, value):
    return options.index(value) if value in options else 0

# ----------------------
# User Inputs
# ----------------------
age = st.slider("Age", 1, 120, values.get("age", 60))

sex_options = ["Male", "Female"]
sex_default = "Male" if values.get("sex", 1) == 1 else "Female"
sex_input = st.radio("Sex", sex_options, index=idx(sex_options, sex_default))
sex = 1 if sex_input == "Male" else 0

cp_options = [0,1,2,3]
cp = st.selectbox("Chest Pain Type", cp_options, index=idx(cp_options, values.get("cp",0)))

trestbps = st.slider("Resting Blood Pressure", 50, 250, values.get("trestbps",120))
chol = st.slider("Cholesterol", 100, 600, values.get("chol",200))

fbs_options = [0,1]
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", fbs_options, index=idx(fbs_options, values.get("fbs",0)))

restecg_options = [0,1,2]
restecg = st.selectbox("Rest ECG", restecg_options, index=idx(restecg_options, values.get("restecg",0)))

thalch = st.slider("Maximum Heart Rate Achieved", 60, 250, values.get("thalch",150))

exang_options = [0,1]
exang = st.radio("Exercise Induced Angina", exang_options, index=idx(exang_options, values.get("exang",0)))

oldpeak = st.slider("Oldpeak", 0.0, 10.0, values.get("oldpeak",0.5), step=0.1)

slope_options = [0,1,2]
slope = st.selectbox("Slope", slope_options, index=idx(slope_options, values.get("slope",1)))

ca_options = [0,1,2,3]
ca = st.selectbox("Number of Major Vessels", ca_options, index=idx(ca_options, values.get("ca",0)))

thal_options = [0,1,2,3]
thal = st.selectbox("Thal", thal_options, index=idx(thal_options, values.get("thal",1)))

# ----------------------
# EXACT MODEL FEATURE ORDER
# IMPORTANT: must match training dataset
# ----------------------
model_feature_order = [
    "id",
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

# ----------------------
# Build full input row
# ----------------------
full_input = {
    "id": 0,
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalch,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

# ----------------------
# Scale required features
# ----------------------
heart_model_array = np.array([[full_input[f] for f in heart_model_features_names]])
heart_model_values = heart_model.transform(heart_model_array)[0]

heart_model_input = full_input.copy()
for i, f in enumerate(heart_model_names):
    heart_model_input[f] = heart_model_values[i]

# ----------------------
# Final input in correct order
# ----------------------
input_final = np.array([[heart_model_input[f] for f in model_feature_order]])

# ----------------------
# Predict
# ----------------------
if st.button("Predict"):

    if input_final.shape[1] != model.n_features_in_:
        st.error(f"Model expects {model.n_features_in_} features but got {input_final.shape[1]}")
        st.stop()

    prediction = model.predict(input_final)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_final)[0][1]
        st.progress(int(prob * 100))
        st.write(f"Risk Probability: {prob*100:.2f}%")

