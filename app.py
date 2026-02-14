import streamlit as st
import numpy as np
import joblib

# 1. Load your trained model
# Ensure 'heart_model.pkl' is in the same folder as this script
model = joblib.load("heart_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict risk level:")

# 2. Define Mappings (Crucial for fixing the "Only Low Risk" bug)
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 3, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 0}
fbs_map = {"True": 1, "False": 0}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}

# 3. User Input UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", list(sex_map.keys()))
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_map.keys()))

with col2:
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_map.keys()))
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", list(thal_map.keys()))

# 4. Prediction Logic
if st.button("Predict Risk Level"):
    # Convert inputs using the maps defined above
    features = np.array([[
        age, 
        sex_map[sex], 
        cp_map[cp], 
        trestbps, 
        chol, 
        fbs_map[fbs], 
        restecg, 
        thalch, 
        exang_map[exang], 
        oldpeak, 
        slope_map[slope], 
        ca, 
        thal_map[thal]
    ]])

    # Make prediction
    prediction = model.predict(features)
    
    st.subheader("Results:")
    if prediction[0] >= 1:
        st.error(f"⚠️ High Risk Detected (Model Output Class: {prediction[0]})")
        st.write("The model indicates clinical signs of heart disease.")
    else:
        st.success("✅ Low Risk Detected")
        st.write("The model indicates a low probability of heart disease.")

    # Debugging: Shows the exact numbers being sent to the model
    with st.expander("See Raw Input Data"):
        st.write(features)



