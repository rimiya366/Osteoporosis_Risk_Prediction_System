import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Osteoporosis Risk Predictor",
    page_icon="🦴",
    layout="wide"
)

# Title and description
st.title("🦴 Osteoporosis Risk Prediction App")
st.write("Enter patient medical details below to evaluate the predicted osteoporosis risk.")

# --- LOAD TRAINED MODEL & SCALER ---
@st.cache_resource
def load_artifacts():
    # Replace these filenames with the exact paths where you saved your model and scaler
    # e.g., joblib.dump(best_gb_model, 'best_gb_model.pkl')
    # e.g., joblib.dump(scaler, 'age_scaler.pkl')
    model = joblib.load('best_gb_model.pkl')
    scaler = joblib.load('age_scaler.pkl')
    return model, scaler

try:
    best_gb_model, scaler = load_artifacts()
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Make sure 'best_gb_model.pkl' and 'age_scaler.pkl' are saved in the same directory.")
    st.stop()

# --- INPUT FORM ---
st.subheader("Patient Clinical Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
    gender = st.selectbox("Gender", options=[("Female", 1), ("Male", 0)], format_func=lambda x: x[0])[1]
    ethnicity = st.selectbox("Race / Ethnicity", options=["Asian", "Caucasian", "Other"])
    body_weight = st.selectbox("Body Weight Category", options=[("Normal / Healthy", 1), ("Low / Underweight", 0)], format_func=lambda x: x[0])[1]
    medical_condition = st.selectbox("Medical Condition", options=["None", "Rheumatoid Arthritis", "Unknown"])

with col2:
    hormonal_changes = st.selectbox("Hormonal Changes", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    family_history = st.selectbox("Family History of Osteoporosis", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    prior_fractures = st.selectbox("Prior Fractures", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    medications = st.selectbox("Taking High-Risk Medications", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

with col3:
    calcium = st.selectbox("Calcium Intake", options=[("Adequate / High", 1), ("Low / Deficient", 0)], format_func=lambda x: x[0])[1]
    vit_d = st.selectbox("Vitamin D Intake", options=[("Adequate / High", 1), ("Low / Deficient", 0)], format_func=lambda x: x[0])[1]
    activity = st.selectbox("Physical Activity Level", options=[("Active", 1), ("Sedentary", 0)], format_func=lambda x: x[0])[1]
    smoking = st.selectbox("Smoking Status", options=[("Non-Smoker", 0), ("Smoker", 1)], format_func=lambda x: x[0])[1]
    alcohol = st.selectbox("Alcohol Consumption", options=[("Non-Drinker / Moderate", 0), ("High", 1)], format_func=lambda x: x[0])[1]

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("Predict Osteoporosis Risk", type="primary", use_container_width=True):
    # 1. One-hot encoding logic for categorical selections
    ethnicity_asian = 1 if ethnicity == "Asian" else 0
    ethnicity_caucasian = 1 if ethnicity == "Caucasian" else 0
    
    medical_ra = 1 if medical_condition == "Rheumatoid Arthritis" else 0
    medical_unknown = 1 if medical_condition == "Unknown" else 0

    # 2. Build DataFrame matching model feature order
    feature_names = [
        'Age', 'Gender', 'Hormonal Changes', 'Family History', 'Body Weight',
        'Calcium Intake', 'Vitamin D Intake', 'Physical Activity', 'Smoking',
        'Alcohol Consumption', 'Medications', 'Prior Fractures',
        'Race/Ethnicity_Asian', 'Race/Ethnicity_Caucasian',
        'Medical Conditions_Rheumatoid Arthritis', 'Medical Conditions_Unknown'
    ]

    patient_data = {
        'Age': age,
        'Gender': gender,
        'Hormonal Changes': hormonal_changes,
        'Family History': family_history,
        'Body Weight': body_weight,
        'Calcium Intake': calcium,
        'Vitamin D Intake': vit_d,
        'Physical Activity': activity,
        'Smoking': smoking,
        'Alcohol Consumption': alcohol,
        'Medications': medications,
        'Prior Fractures': prior_fractures,
        'Race/Ethnicity_Asian': ethnicity_asian,
        'Race/Ethnicity_Caucasian': ethnicity_caucasian,
        'Medical Conditions_Rheumatoid Arthritis': medical_ra,
        'Medical Conditions_Unknown': medical_unknown
    }

    input_df = pd.DataFrame([patient_data], columns=feature_names)

    # 3. Scale ONLY the Age feature
    input_df[['Age']] = scaler.transform(input_df[['Age']])

    # 4. Predict
    prediction = best_gb_model.predict(input_df)[0]
    probabilities = best_gb_model.predict_proba(input_df)[0]
    high_risk_prob = probabilities[1] * 100

    # --- DISPLAY RESULTS ---
    st.subheader("Assessment Output")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK DETECTED")
        else:
            st.success("✅ LOW RISK DETECTED")
            
    with res_col2:
        st.metric(label="Osteoporosis Probability", value=f"{high_risk_prob:.2f}%")
        st.progress(high_risk_prob / 100)

    if prediction == 1:
        st.warning("Recommendation: Clinical evaluation and Bone Mineral Density (BMD / DEXA) scanning recommended.")
    else:
        st.info("Recommendation: Maintain a balanced diet rich in Calcium and Vitamin D, along with regular weight-bearing exercise.")