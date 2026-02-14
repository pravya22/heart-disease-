import streamlit as st
import numpy as np
import joblib

# 1. LOAD THE MODEL
# Ensure your file is named 'heart_model.pkl' and is in the same folder
try:
    model = joblib.load("heart_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'heart_model.pkl' is uploaded.")

# 2. APP CONFIGURATION
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")
st.markdown("---")

# 3. DEFINE MAPPINGS
# This converts the readable text in the app to the numbers (0, 1, 2, 3) the model needs.
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 3, "Atypical Angina": 1, "Non-Anginal": 2, "Asymptomatic": 0}
fbs_map = {"True": 1, "False": 0}
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"Yes": 1, "No": 0}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}

# 4. USER INTERFACE (SIDE-BY-SIDE COLUMNS)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", list(sex_map.keys()))
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
    chol = st.number_input("Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_map.keys()))
    restecg = st.selectbox("Resting ECG Results", list(restecg_map.keys()))

with col2:
    thalch = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina", list(exang_map.keys()))
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, step=0.1, value=0.0)
    slope = st.selectbox("Slope of Peak Exercise ST", list(slope_map.keys()))
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia Status", list(thal_map.keys()))

st.markdown("---")

# 5. PREDICTION LOGIC
if st.button("Predict Risk Status"):
    # Create the feature array - NO 'id_value' included here!
    # The order MUST match the columns your model was trained on.
    final_input = np.array([[
        age, 
        sex_map[sex], 
        cp_map[cp], 
        trestbps, 
        chol, 
        fbs_map[fbs], 
        restecg_map[restecg], 
        thalch, 
        exang_map[exang], 
        oldpeak, 
        slope_map[slope], 
        ca, 
        thal_map[thal]
    ]])

    # Make the prediction
    prediction = model.predict(final_input)
    
    st.subheader("Final Result:")
    
    # In the UCI dataset, 0 is 'Healthy' and 1-4 represent 'Heart Disease'
    if prediction[0] >= 1:
        st.error(f"⚠️ HIGH RISK: The model predicts heart disease (Risk Level {prediction[0]})")
        st.info("Clinical indicators like high ST depression (oldpeak) or asymptomatic chest pain influenced this result.")
    else:
        st.success("✅ LOW RISK: The model indicates no significant heart disease.")

    # OPTIONAL: Debugging to confirm 13 features are being sent
    with st.expander("Technical Details (Feature Check)"):
        st.write(f"Number of features sent to model: {final_input.shape[1]}")
        st.write("Processed Input Array:", final_input)

