# streamlit_app/Home.py

import os
import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
import shap
from streamlit_shap import st_shap

# Config
st.set_page_config(page_title="Client Satisfaction Dashboard", layout="wide")
st.title("HIV Client Satisfaction Dashboard")
st.markdown("Enter client details for AI-powered prediction & explanation. The user interface will collect raw data and calculate the necessary features for the model automatically.")

# Load API base URL from env or default to localhost
API_BASE = os.getenv("API_URL", "http://localhost:7860")

# =============================================================================
# ✅ [NEW] Artifacts and Mappings
# =============================================================================
@st.cache_resource
def load_artifacts():
    """Load all necessary model and feature artifacts."""
    try:
        model_features = joblib.load("model/important_features.joblib")
        categories = joblib.load("model/categories.joblib")
        return model_features, categories
    except FileNotFoundError:
        st.error("Could not load model artifacts. Make sure 'important_features.joblib' and 'categories.joblib' are in the 'model/' directory.")
        return None, None

model_features, categories = load_artifacts()

# (b) Define Likert scale options
likert_map = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neither Agree Or Disagree': 3,
    'Agree': 4,
    'Strongly Agree': 5
}
likert_options = list(likert_map.keys())

# =============================================================================
# ✅ [NEW] UI and Feature Engineering Logic
# =============================================================================
if model_features:
    st.subheader("Client & Provider Interaction Details")
    
    with st.form("predict_form"):
        raw_inputs = {}
        st.markdown("#### Patient Demographics")
        demo_cols = st.columns(3)
        raw_inputs['Age'] = demo_cols[0].number_input("Age", min_value=0, value=35)
        raw_inputs['Employment_Grouped'] = demo_cols[1].selectbox("Employment Status", categories.get('Employment_Grouped', ['Employed', 'Unemployed']))
        raw_inputs['State'] = demo_cols[2].selectbox("State", categories.get('State', ['State A', 'State B']))

        st.markdown("#### Care Duration (in Years)")
        care_cols = st.columns(3)
        # (c) Collect raw duration features
        raw_inputs['HIV_Duration_Years'] = care_cols[0].number_input("Duration of HIV Diagnosis (Years)", min_value=0.0, value=5.0, format="%.1f")
        raw_inputs['Care_Duration_Years'] = care_cols[1].number_input("Duration at Current Facility (Years)", min_value=0.0, value=2.0, format="%.1f")
        raw_inputs['Facility_Care_Dur_Years'] = care_cols[2].number_input("Total Duration of Care (All Facilities, Years)", min_value=0.0, value=5.0, format="%.1f")

        st.markdown("#### Provider Interaction Scores (Rate 1-5)")
        interaction_cols = st.columns(3)
        # (d) Collect raw scores for interaction features
        raw_inputs['Empathy_Score'] = interaction_cols[0].slider("Average Empathy Score", 1.0, 5.0, 4.0)
        raw_inputs['Listening_Score'] = interaction_cols[1].slider("Average Listening Score", 1.0, 5.0, 4.0)
        raw_inputs['Decision_Share_Score'] = interaction_cols[2].slider("Average Decision Sharing Score", 1.0, 5.0, 3.0)
        
        st.markdown("#### Communication & Information")
        comm_cols = st.columns(2)
        # (b) Use selectbox for Likert scale questions
        raw_inputs['Exam_Explained'] = comm_cols[0].selectbox("The provider explained exams/procedures clearly.", options=likert_options, index=3)
        raw_inputs['Discuss_NextSteps'] = comm_cols[1].selectbox("The provider discussed the next steps in my care.", options=likert_options, index=3)

        submitted = st.form_submit_button("Predict Client Satisfaction")

    if submitted:
        # =====================================================================
        # ✅ [NEW] On-the-fly Feature Engineering
        # =====================================================================
        final_features = {}
        
        # (c) Calculate HIV Care Duration Ratio
        # Adding 0.1 to denominator to avoid division by zero, as in the original code
        final_features['HIV_Care_Duration_Ratio'] = raw_inputs['HIV_Duration_Years'] / (raw_inputs['Care_Duration_Years'] + 0.1)
        
        # (d) Calculate Interaction Features
        final_features['Empathy_Listening_Interaction'] = raw_inputs['Empathy_Score'] * raw_inputs['Listening_Score']
        final_features['Empathy_DecisionShare_Interaction'] = raw_inputs['Empathy_Score'] * raw_inputs['Decision_Share_Score']
        
        # Map Likert scale text to numeric values
        final_features['Exam_Explained'] = likert_map[raw_inputs['Exam_Explained']]
        final_features['Discuss_NextSteps'] = likert_map[raw_inputs['Discuss_NextSteps']]

        # Add other features that are directly used by the model
        direct_features = ['Age', 'Employment_Grouped', 'State', 'Facility_Care_Dur_Years', 'Education_Grouped']
        for feature in direct_features:
            if feature in raw_inputs:
                final_features[feature] = raw_inputs[feature]
            elif feature in categories: # Handle features that might not have a direct input but are categorical
                 final_features[feature] = categories[feature][0] # Default to the first category
            else:
                 final_features[feature] = 0 # Default numeric to 0

        # Ensure all required model features are present, adding placeholders if necessary
        for feature in model_features:
            if feature not in final_features:
                final_features[feature] = 0 # Default any other missing features to 0

        st.write("### Calculated Features Sent to API")
        st.json({k: (f"{v:.2f}" if isinstance(v, float) else v) for k,v in final_features.items()})

        # =====================================================================
        # API Call and Display Logic (largely unchanged)
        # =====================================================================
        try:
            resp = requests.post(f"{API_BASE}/predict", json={"features": final_features})
            if resp.status_code == 200:
                res = resp.json()

                st.success(f"**Prediction:** {res['prediction']} (Confidence {res['confidence']})")
                
                st.subheader("Top 3 Contributing Factors (SHAP)")
                st.json(res.get("top_features", {}))

                if res.get("suggestions"):
                    st.info(f"**AI Suggestions:** {res['suggestions']}")

                st.subheader("Full GenAI Explanation")
                st.markdown(res.get("genai_explanation", "No explanation provided."))

                st.subheader("SHAP Visualization")
                try:
                    shap_vals = np.array(res.get("shap_values", []))
                    if shap_vals.size == 0:
                        st.warning("No SHAP values returned from API.")
                    else:
                        # Use the final_features sent to the API for the plot
                        feature_names = list(final_features.keys())
                        feature_values = np.array(list(final_features.values()))

                        explanation = shap.Explanation(
                            values=shap_vals,
                            base_values=res.get("shap_base_value", 0), 
                            data=feature_values,
                            feature_names=feature_names
                        )
                        st_shap(shap.plots.waterfall(explanation[0]), height=400)

                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {e}")

            else:
                st.error(f"API Error ({resp.status_code}): {resp.json().get('detail', resp.text)}")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection or API error: {e}")

    st.divider()
    st.subheader("Prediction Log")
    try:
        logs = requests.get(f"{API_BASE}/logs").json()
        if logs:
            logs_df = pd.DataFrame(logs)
            st.dataframe(logs_df[["instance_idx", "prediction", "confidence"]])

            if "suggestions" in logs_df.columns:
                st.write("AI Suggestions from logs:")
                st.write(logs_df["suggestions"])
            else:
                st.warning("No suggestions found in logs.")
        else:
            st.info("No logs yet.")
    except Exception as e:
        st.warning(f"Could not load logs: {e}")