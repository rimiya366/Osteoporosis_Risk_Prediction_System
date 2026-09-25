import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Osteoporosis Risk Assessor",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Asset Loader Function
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    # Find root project directory (one level up from app/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Path to models folder
    model_path = os.path.join(project_root, 'models', 'osteoporosis_gb_pipeline.pkl')
    
    # Fallback check: in case model is placed directly in app/ directory
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, 'osteoporosis_gb_pipeline.pkl')

    if not os.path.exists(model_path):
        st.error(f"Missing model file at path: {model_path}")
        return None, None

    try:
        loaded_object = joblib.load(model_path)
        if isinstance(loaded_object, dict):
            return loaded_object.get('model'), loaded_object.get('feature_names')
        return loaded_object, getattr(loaded_object, 'feature_names_in_', None)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None, None

model, expected_features = load_pipeline()

# ---------------------------------------------------------
# 3. Header & Sidebar UI
# ---------------------------------------------------------
st.title("🦴 Osteoporosis Risk Prediction System")
st.markdown("""
This clinical decision-support tool utilizes a **Gradient Boosting Classifier** trained on patient demographic and medical profile data to estimate osteoporosis risk.
""")

st.sidebar.header("📋 Patient Clinical Profile")

if model is not None:
    # Input controls
    age = st.sidebar.slider("Age", min_value=18, max_value=95, value=50, step=1)
    gender = st.sidebar.selectbox("Gender", options=["Female", "Male"])
    hormonal_changes = st.sidebar.selectbox("Hormonal Changes", options=["Normal", "Postmenopausal"])
    family_history = st.sidebar.selectbox("Family History of Osteoporosis", options=["No", "Yes"])
    race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", options=["Caucasian", "Asian", "African American"])
    body_weight = st.sidebar.selectbox("Body Weight", options=["Normal", "Underweight"])
    calcium_intake = st.sidebar.selectbox("Calcium Intake", options=["Low", "Adequate"])
    vitamin_d_intake = st.sidebar.selectbox("Vitamin D Intake", options=["Sufficient", "Insufficient"])
    physical_activity = st.sidebar.selectbox("Physical Activity", options=["Active", "Sedentary"])
    smoking = st.sidebar.selectbox("Smoking Status", options=["No", "Yes"])
    alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption", options=["None", "Moderate"])
    medical_conditions = st.sidebar.selectbox("Medical Conditions", options=["None", "Hyperthyroidism", "Rheumatoid Arthritis"])
    medications = st.sidebar.selectbox("Medications", options=["None", "Corticosteroids"])
    prior_fractures = st.sidebar.selectbox("Prior Fractures", options=["No", "Yes"])

    # Raw user inputs matched to Jupyter training column names
    raw_input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Hormonal Changes': hormonal_changes,
        'Family History': family_history,
        'Race/Ethnicity': race_ethnicity,
        'Body Weight': body_weight,
        'Calcium Intake': calcium_intake,
        'Vitamin D Intake': vitamin_d_intake,
        'Physical Activity': physical_activity,
        'Smoking': smoking,
        'Alcohol Consumption': alcohol_consumption,
        'Medical Conditions': medical_conditions,
        'Medications': medications,
        'Prior Fractures': prior_fractures
    }])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Selected Patient Parameters")
        st.dataframe(raw_input_data.T.rename(columns={0: "Value"}), use_container_width=True)

    with col2:
        st.subheader("Prediction Analysis")
        
        try:
            # Check if model requires raw features or explicit dummy encoding
            if hasattr(model, 'named_steps') or expected_features is None:
                # Full pipeline handling internal encoding
                probabilities = model.predict_proba(raw_input_data)
            else:
                # Pre-encoded model fallback
                encoded_df = pd.get_dummies(raw_input_data)
                aligned_df = encoded_df.reindex(columns=expected_features, fill_value=0)
                probabilities = model.predict_proba(aligned_df)

            risk_probability = float(probabilities[0, 1])

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Ensure input column names match the exact column names used in your Jupyter notebook.")
            risk_probability = 0.0

        threshold = st.slider("Clinical Decision Threshold", min_value=0.20, max_value=0.80, value=0.50, step=0.05)
        predicted_class = 1 if risk_probability >= threshold else 0

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_probability * 100,
            number={'suffix': "%"},
            title={'text': "Osteoporosis Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2C3E50"},
                'steps': [
                    {'range': [0, threshold * 100], 'color': "#2ECC71"},
                    {'range': [threshold * 100, 100], 'color': "#E74C3C"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if predicted_class == 1:
            st.error(f"⚠️ **HIGH RISK DETECTED** (Probability: {risk_probability:.1%})")
            st.warning("Recommendation: Diagnostic Bone Mineral Density (BMD) testing advised.")
        else:
            st.success(f"✅ **LOW RISK DETECTED** (Probability: {risk_probability:.1%})")
            st.info("Recommendation: Maintain routine health monitoring.")

else:
    st.error("Model assets file (`osteoporosis_gb_pipeline.pkl`) not found in `models/` directory.")