# streamlit_app/Home.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import json
import requests
import logging
from dotenv import load_dotenv
from pathlib import Path

# --- Configuration and Logging ---
st.set_page_config(
    page_title="Client Satisfaction Dashboard",
    page_icon="https://static.vecteezy.com/system/resources/previews/013/695/803/original/customer-satisfaction-icon-style-free-vector.jpg",
    initial_sidebar_state="expanded",
    layout="wide"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("streamlit_app")

# Load environment variables for API key
load_dotenv(dotenv_path=Path('.') / '.env')

# --- Global Artifacts and Mappings ---
label_map = {
    0: 'Very Dissatisfied',
    1: 'Dissatisfied',
    2: 'Neutral',
    3: 'Satisfied',
    4: 'Very Satisfied'
}

likert_map = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neither Agree Or Disagree': 3,
    'Agree': 4,
    'Strongly Agree': 5
}
likert_options = list(likert_map.keys())

# --- Caching and Artifact Loading ---
@st.cache_resource
def load_artifacts():
    """Load all necessary model and feature artifacts safely."""
    try:
        model = joblib.load("model/top10_model.joblib")
        top_features = joblib.load("model/important_features.joblib")
        categories = joblib.load("model/categories.joblib")
        label_encoder = joblib.load("model/label_encoder.joblib")
        logger.info("Model, features, categories, and encoder loaded successfully.")
        return model, top_features, categories, label_encoder
    except FileNotFoundError as e:
        st.error(f"Could not load model artifacts: {e}. Make sure all `.joblib` files are in the 'model/' directory.")
        logger.error(f"Failed to load model artifacts: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading artifacts: {e}")
        logger.error(f"Unexpected error loading artifacts: {e}")
        return None, None, None, None

model, top_features, categories, label_encoder = load_artifacts()

# --- Helper Functions ---
def enforce_categorical_dtypes(df, categorical_cols):
    """Ensures specified columns in DataFrame are of 'category' dtype."""
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

@st.cache_resource
def get_shap_explainer(_model):
    """Returns a cached SHAP TreeExplainer for the given model."""
    return shap.TreeExplainer(_model)

# --- Rule-Based Logic Engine ---
def rule_overall_client_satisfaction(instance_data):
    """Evaluates multiple features to generate a comprehensive picture of satisfaction."""
    reasons, suggestions = [], []
    if instance_data.get('Empathy_Listening_Interaction', 15) < 9:
        reasons.append("Low empathy and poor listening from the provider likely reduced satisfaction.")
        suggestions.append("Train providers to improve empathy and active listening skills.")
    if instance_data.get('Empathy_DecisionShare_Interaction', 15) < 9:
        reasons.append("A perceived lack of empathy or poor shared decision-making contributed to dissatisfaction.")
        suggestions.append("Ensure clients feel their opinions are valued and they are included in care planning.")
    if instance_data.get('Exam_Explained', 3) < 3:
        reasons.append("Medical exams or procedures were not explained clearly to the client.")
        suggestions.append("Standardize communication protocols to improve clarity around clinical procedures.")
    return len(reasons) > 0, reasons, suggestions

RULES = [
    (
        'Overall Client Satisfaction',
        "Address communication, education, employment, treatment duration, and participatory care.",
        (rule_overall_client_satisfaction, True)
    )
]

# --- Generative AI Explanation Engine ---
def generate_ai_explanation(prediction, confidence, top_shap_features, reasons, suggestions, openrouter_api_key):
    """Generates a detailed, structured explanation using a Generative AI model."""
    if not openrouter_api_key:
        return "GenAI explanation unavailable: API key not configured."
    prompt = f"""
    You are an expert AI Data Analyst for a clinical quality improvement team. Your task is to explain a client satisfaction prediction in a clear, actionable way.

    **Client Context:**
    - **Setting:** HIV Clinic
    - **Goal:** Understand drivers of client satisfaction to improve care quality.
    - **Prediction:** The model predicts this client's satisfaction level is **'{prediction}'**.
    - **Confidence:** The model is **{confidence}** confident in this prediction.

    **AI & Rule-Based Analysis Results:**
    1.  **Top Quantitative Drivers (from SHAP model analysis):**
        ```json
        {json.dumps(top_shap_features, indent=2)}
        ```
    2.  **Qualitative Insights (from clinical rules):**
        - **Identified Issues/Reasons:** {"- " + "\n- ".join(reasons) if reasons else "None."}
        - **System Suggestions:** {"- " + "\n- ".join(suggestions) if suggestions else "None."}

    **Your Task:** Structure your response in three distinct sections using markdown:
    ### 1. Executive Summary
    Provide a one-paragraph robust summary of the prediction and the primary reasons behind it.
    ### 2. Analysis of Drivers
    Explain *how* the top quantitative drivers and the qualitative insights connect. Translate feature names (e.g., 'Empathy_Listening_Interaction') into plain language.
    ### 3. Actionable Recommendations
    List 2-3 concrete, practical steps the clinical team can take based on this specific client's feedback."""
    
    headers = {"Authorization": f"Bearer {openrouter_api_key}", "Content-Type": "application/json"}
    body = {"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(body), timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logger.error(f"GenAI API request failed: {e}")
        return f"Error connecting to GenAI service: {e}."
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected GenAI API response format: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
        return "Error parsing GenAI service response."
    except Exception as e:
        logger.error(f"An unexpected error occurred in GenAI explanation: {e}")
        return f"An unexpected error occurred during GenAI explanation: {e}"
    
# --- Main Prediction and Explanation Pipeline ---
def explain_prediction_integrated(model, X_instance_df, openrouter_api_key, categorical_cols):
    """Generates prediction and a full explanation for a single instance."""
    if X_instance_df.shape[0] != 1:
        raise ValueError("Input DataFrame must contain exactly one instance for explanation.")

    instance = enforce_categorical_dtypes(X_instance_df.copy(), categorical_cols)

    # --- Prediction ---
    preds_proba = model.predict_proba(instance)[0]
    pred_class = np.argmax(preds_proba)
    confidence = f"{round(float(np.max(preds_proba)) * 100, 1)}%"
    mapped_pred = label_map.get(int(pred_class), "Unknown")

    # --- SHAP Value Calculation ---
    explainer = get_shap_explainer(model)
    shap_values_raw = explainer.shap_values(instance)
    expected_value_raw = explainer.expected_value

    # ✅ FIX: Robustly handle different multi-class SHAP output formats to resolve the ValueError.
    # This ensures we get a 1D array of SHAP values for the predicted class.
    if isinstance(shap_values_raw, list):
        # Format: list of arrays, one per class. Array shape: (n_samples, n_features)
        shap_values_for_class = shap_values_raw[pred_class][0]
        base_value_for_class = expected_value_raw[pred_class]
    else:
        # Format: 3D numpy array. Shape: (n_samples, n_features, n_classes)
        shap_values_for_class = shap_values_raw[0, :, pred_class]
        base_value_for_class = expected_value_raw[pred_class]

    # Now, `shap_values_for_class` is a 1D array, so sorting will work.
    shap_dict = dict(zip(instance.columns, shap_values_for_class))
    top_shap_features = {k: round(float(v), 3) for k, v in sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:3]}

    # --- Rule-Based Analysis ---
    instance_data = instance.iloc[0].to_dict()
    reasons, suggestions = [], []
    for _, _, rule_tuple in RULES:
        rule_fn, _ = rule_tuple
        is_triggered, rule_reasons, rule_suggestions = rule_fn(instance_data)
        if is_triggered:
            reasons.extend(rule_reasons)
            suggestions.extend(rule_suggestions)

    # --- Generative AI Synthesis ---
    genai_explanation = generate_ai_explanation(
        mapped_pred, confidence, top_shap_features, reasons, suggestions, openrouter_api_key
    )

    return {
        'prediction': mapped_pred,
        'confidence': confidence,
        'top_features': top_shap_features,
        'reasons': reasons,
        'suggestions': suggestions,
        'genai_explanation': genai_explanation,
        'shap_values': shap_values_for_class,
        'shap_base_value': base_value_for_class,
        'feature_values': instance.iloc[0].values,
        'feature_names': list(instance.columns)
    }

# --- Streamlit User Interface ---
# Centering and coloring the main title
st.markdown(f"""
            <h1 style='text-align: center; color: #FF4B4B; font-style: italic;'>
                <img src='https://static.vecteezy.com/system/resources/previews/013/695/803/original/customer-satisfaction-icon-style-free-vector.jpg' style='vertical-align: middle; height: 50px; margin-right: 10px;'>
                HIV Client Satisfaction Dashboard         
           </h1>
            """, unsafe_allow_html=True
            )
#st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>HIV Client Satisfaction Dashboard</h1>", unsafe_allow_html=True)
# Centering and coloring the descriptive text
st.markdown("<p style='text-align: center; color: #6C3483; font-size: 1.8em;'>Enter client details for an AI-powered prediction and explanation. The interface automatically calculates the necessary features for the model.</p>",
    unsafe_allow_html=True)

if all([model, top_features, categories, label_encoder]):
    st.divider()
    with st.form("predict_form"):
        st.subheader("📝 Client & Provider Interaction Details")
        raw_inputs = {}

        st.markdown("##### Patient Demographics")
        demo_cols = st.columns(3)
        raw_inputs['Age'] = demo_cols[0].number_input("Age", min_value=18, max_value=100, value=35)
        raw_inputs['Employment_Grouped'] = demo_cols[1].selectbox("Employment Status", categories.get('Employment_Grouped', []))
        raw_inputs['Education_Grouped'] = demo_cols[2].selectbox("Education Level", categories.get('Education_Grouped', []))
        if 'State' in top_features:
            raw_inputs['State'] = demo_cols[0].selectbox("State", categories.get('State', []), key="state_select")

        st.markdown("##### Care History (in Years)")
        care_cols = st.columns(3)
        raw_inputs['HIV_Duration_Years'] = care_cols[0].number_input("Duration of HIV Diagnosis", min_value=0.0, value=5.0, step=0.5, format="%.1f")
        raw_inputs['Care_Duration_Years'] = care_cols[1].number_input("Duration at This Facility", min_value=0.0, value=2.0, step=0.5, format="%.1f")
        raw_inputs['Facility_Care_Dur_Years'] = care_cols[2].number_input("Total Duration in Care (All Facilities)", min_value=0.0, value=5.0, step=0.5, format="%.1f")

        st.markdown("##### Provider Interaction Scores (Rate 1-5)")
        interaction_cols = st.columns(3)
        raw_inputs['Empathy_Score'] = interaction_cols[0].slider("Provider Empathy Score", 1.0, 5.0, 4.0, 0.5)
        raw_inputs['Listening_Score'] = interaction_cols[1].slider("Provider Listening Score", 1.0, 5.0, 4.0, 0.5)
        raw_inputs['Decision_Share_Score'] = interaction_cols[2].slider("Shared Decision-Making Score", 1.0, 5.0, 3.0, 0.5)

        st.markdown("##### Communication & Information")
        comm_cols = st.columns(2)
        raw_inputs['Exam_Explained'] = comm_cols[0].selectbox("Provider explained exams/procedures clearly.", options=likert_options, index=3)
        raw_inputs['Discuss_NextSteps'] = comm_cols[1].selectbox("Provider discussed the next steps in my care.", options=likert_options, index=3)

        submitted = st.form_submit_button("🚀 Predict & Explain Satisfaction")

    if submitted:
        with st.spinner("Analyzing data and generating explanation..."):
            final_features = {}
            final_features['HIV_Care_Duration_Ratio'] = raw_inputs['HIV_Duration_Years'] / (raw_inputs['Care_Duration_Years'] + 0.1)
            final_features['Empathy_Listening_Interaction'] = raw_inputs['Empathy_Score'] * raw_inputs['Listening_Score']
            final_features['Empathy_DecisionShare_Interaction'] = raw_inputs['Empathy_Score'] * raw_inputs['Decision_Share_Score']
            final_features['Exam_Explained'] = likert_map[raw_inputs['Exam_Explained']]
            final_features['Discuss_NextSteps'] = likert_map[raw_inputs['Discuss_NextSteps']]

            input_data = {}
            for feature in top_features:
                if feature in final_features:
                    input_data[feature] = final_features[feature]
                elif feature in raw_inputs:
                    input_data[feature] = raw_inputs[feature]
                else:
                    if feature in categories:
                        input_data[feature] = categories[feature][0]
                    else:
                        input_data[feature] = 0.0

            input_df = pd.DataFrame([input_data], columns=top_features)
            categorical_features_in_model = [col for col in top_features if col in categories]
            openrouter_api_key = os.getenv("SATISFACTION_APP_KEY")

            try:
                result = explain_prediction_integrated(
                    model=model, X_instance_df=input_df,
                    openrouter_api_key=openrouter_api_key,
                    categorical_cols=categorical_features_in_model
                )

                st.divider()
                st.header("Prediction Results")
                pred_col, conf_col = st.columns(2)
                pred_col.metric(label="Predicted Satisfaction Level", value=result['prediction'])
                conf_col.metric(label="Prediction Confidence", value=result['confidence'])

                tab1, tab2, tab3 = st.tabs(["✨ **AI Summary**", "🧠 **Detailed Analysis**", "⚙️ **Model Input**"])

                with tab1:
                    st.subheader("📝 Generative AI Explanation")
                    st.markdown(result.get("genai_explanation", "No explanation provided."))

                with tab2:
                    st.subheader("🎯 Top 3 Contributing Factors (SHAP)")
                    for feature, value in result.get("top_features", {}).items():
                        impact = "🟢 Positive" if value > 0 else "🔴 Negative"
                        st.markdown(f"- **{feature.replace('_', ' ')}**: SHAP Value = `{value:.3f}` ({impact} impact)")

                    st.subheader("📈 SHAP Waterfall Plot")
                    explanation_obj = shap.Explanation(
                        values=result['shap_values'], base_values=result['shap_base_value'],
                        data=result['feature_values'], feature_names=result['feature_names']
                    )
                    st_shap(shap.plots.waterfall(explanation_obj), height=400, width=1100)

                with tab3:
                    st.subheader("🔢 Features Sent to Model")
                    st.json({k: (f"{v:.2f}" if isinstance(v, (float, np.floating)) else v) for k, v in input_df.iloc[0].to_dict().items()})

            except Exception as e:
                st.error(f"An error occurred during prediction or explanation: {e}")
                logger.exception(f"Error during prediction or explanation: {e}")
else:
    st.warning("Application artifacts could not be loaded. The app cannot proceed.")