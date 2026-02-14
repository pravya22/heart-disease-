import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("heart_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Adjust the sliders or selections to test Low Risk and High Risk scenarios:")

# ----------------------
# User Inputs
# ----------------------
age = st.slider("Age", 1, 120, 60)  # middle value
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

# ----------------------
# Encode categorical
# ----------------------
sex = 1 if sex == "Male" else 0

# One-hot encode Chest Pain
cp_0 = 1 if cp == 0 else 0
cp_1 = 1 if cp == 1 else 0
cp_2 = 1 if cp == 2 else 0
cp_3 = 1 if cp == 3 else 0

# One-hot encode Thal
thal_0 = 1 if thal == 0 else 0
thal_1 = 1 if thal == 1 else 0
thal_2 = 1 if thal == 2 else 0
thal_3 = 1 if thal == 3 else 0

# ----------------------
# Prepare input
# ----------------------
id_value = 0
input_data = np.array([[
    id_value, age, sex,
    cp_0, cp_1, cp_2, cp_3,
    trestbps, chol, fbs, restecg, thalch, exang, oldpeak,
    slope, ca,
    thal_0, thal_1, thal_2, thal_3
]])

# ----------------------
# Predict
# ----------------------
if st.button("Predict"):
    # Check feature count
    if input_data.shape[1] != model.n_features_in_:
        st.error(f"⚠️ Feature count mismatch! "
                 f"Model expects {model.n_features_in_}, input has {input_data.shape[1]}.")
    else:
        prediction = model.predict(input_data)

        # Risk display
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        # Show probability with bar
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]
            st.progress(int(proba*100))
            st.write(f"Risk Probability: {proba*100:.2f}%")



