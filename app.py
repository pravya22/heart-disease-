import streamlit as st
import numpy as np
import joblib
import os

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Use the sliders to test Low Risk and High Risk scenarios.")

# ----------------------
# Load model and scaler safely
# ----------------------
MODEL_PATH = "heart_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler file not found. Upload them to GitHub repo.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------------------
# Handle scaler feature names safely
# ----------------------
if hasattr(scaler, "feature_names_in_"):
    scaled_feature_names = list(scaler.feature_names_in_)
else:
    # fallback if attribute missing
    scaled_feature_names = ["age", "trestbps", "chol", "thalch", "oldpeak"]

# ----------------------
# Preset values
# ----------------------
high_risk = {
    "age": 65, "sex": 1, "cp": 3, "trestbps": 180, "chol": 300,
    "fbs": 1, "restecg": 1, "thalch": 100, "exang": 1,
    "oldpeak": 4.0, "slope": 2, "ca": 3, "thal": 2
}

low_risk = {
    "age": 40, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
    "fbs": 0, "restecg": 0, "thalch": 170, "exang": 0,
    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 1
}

preset = st.radio("Select Preset", ["Custom", "High Risk", "Low Risk"])

if preset == "High Risk":
    values = high_risk
elif preset == "Low Risk":
    values = low_risk
else:
    values = {}

# ----------------------
# Helper function for selectbox index
# ----------------------
def get_index(options, value):
    return options.index(value) if value in options else 0

# ----------------------
# User inputs
# ----------------------
age = st.slider("Age", 1, 120, values.get("age", 60))

sex_options = ["Male", "Female"]
sex_default = "Male" if values.get("sex", 1) == 1 else "Female"
sex_input = st.radio("Sex", sex_options, index=get_index(sex_options, sex_default))
sex = 1 if sex_input == "Male" else 0

cp_options = [0,1,2,3]
cp = st.selectbox("Chest Pain Type", cp_options,
                  index=get_index(cp_options, values.get("cp",0)))

trestbps = st.slider("Resting Blood Pressure", 50, 250, values.get("trestbps",120))
chol = st.slider("Cholesterol", 100, 600, values.get("chol",200))

fbs_options = [0,1]
fbs = st.radio("Fasting Blood Sugar >120", fbs_options,
               index=get_index(fbs_options, values.get("fbs",0)))

restecg_options = [0,1,2]
restecg = st.selectbox("Rest ECG", restecg_options,
                       index=get_index(restecg_options, values.get("restecg",0)))

thalch = st.slider("Max Heart Rate", 60, 250, values.get("thalch",150))

exang_options = [0,1]
exang = st.radio("Exercise Induced Angina", exang_options,
                 index=get_index(exang_options, values.get("exang",0)))

oldpeak = st.slider("Oldpeak", 0.0, 10.0, values.get("oldpeak",0.5), step=0.1)

slope_options = [0,1,2]
slope = st.selectbox("Slope", slope_options,
                     index=get_index(slope_options, values.get("slope",1)))

ca_options = [0,1,2,3]
ca = st.selectbox("Major Vessels", ca_options,
                  index=get_index(ca_options, values.get("ca",0)))

thal_options = [0,1,2,3]
thal = st.selectbox("Thal", thal_options,
                    index=get_index(thal_options, values.get("thal",1)))

# ----------------------
# Prepare input dictionary
# ----------------------
input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalch": thalch,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}

# ----------------------
# Scale selected features
# ----------------------
scaled_values = np.array([[input_dict[f] for f in scaled_feature_names]])
scaled_values = scaler.transform(scaled_values)

remaining_features = [f for f in input_dict if f not in scaled_feature_names]
remaining_values = np.array([[input_dict[f] for f in remaining_features]])

# if model trained with ID column keep dummy
id_value = np.array([[0]])

input_final = np.hstack([id_value, scaled_values, remaining_values])

# ----------------------
# Prediction
# ----------------------
if st.button("Predict"):

    if input_final.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch. Model expects {model.n_features_in_} but got {input_final.shape[1]}")
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
