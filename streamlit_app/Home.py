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
st.set_page_config(page_title="Client Satisfaction Dashboard", layout="wide")
st.title("HIV Client Satisfaction Dashboard")
st.markdown("Enter client details for AI-powered prediction & explanation. The user interface will collect raw data and calculate the necessary features for the model automatically.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log") # Log to a file within the container
    ]
)
logger = logging.getLogger("streamlit_app")

# Load environment variables (for API key)
# Adjust path if .env is in project root, e.g., Path('.') / '.env'
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

@st.cache_resource
def load_artifacts():
    """Load all necessary model and feature artifacts."""
    try:
        model = joblib.load("model/top10_model.joblib")
        top_features = joblib.load("model/important_features.joblib")
        categories = joblib.load("model/categories.joblib")
        label_encoder = joblib.load("model/label_encoder.joblib")

        logger.info("Model, features, categories, and encoder loaded successfully.")
        return model, top_features, categories, label_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Could not load model artifacts: {e}. Make sure all .joblib files are in the 'model/' directory.")
        logger.error(f"Failed to load model artifacts: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading artifacts: {e}")
        logger.error(f"Unexpected error loading artifacts: {e}")
        return None, None, None, None

model, top_features, categories, label_encoder = load_artifacts()

# --- Helper Functions (Moved/Adapted from explanation_engine.py and model_utils.py) ---

def enforce_categorical_dtypes(df, categorical_cols):
    """Ensures specified columns in DataFrame are of 'category' dtype."""
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def get_shap_explainer(model):
    """Returns a SHAP TreeExplainer for the given model."""
    return shap.TreeExplainer(model)

# Rule-Based Logic (from explanation_engine.py) - Keep this here for direct use
def rule_overall_client_satisfaction(instance_data):
    """
    Evaluates multiple new top features to generate a comprehensive picture of satisfaction.
    """
    reasons, suggestions = [], []

    # Empathy + Listening interaction
    if instance_data.get('Empathy_Listening_Interaction', 15) < 9:
        reasons.append("Low empathy and poor listening likely reduced satisfaction.")
        suggestions.append("Train providers to improve empathy and active listening.")
    elif instance_data.get('Empathy_Listening_Interaction', 15) > 15:
        reasons.append("Strong empathy and active listening boosted client satisfaction.")
        suggestions.append("Encourage continued focus on empathetic listening.")

    # Empathy + Decision-sharing interaction
    if instance_data.get('Empathy_DecisionShare_Interaction', 15) < 9:
        reasons.append("Lack of empathy or poor decision-sharing contributed to dissatisfaction.")
        suggestions.append("Ensure clients feel heard and included in their care planning.")
    elif instance_data.get('Empathy_DecisionShare_Interaction', 15) > 15:
        reasons.append("Clients felt supported and involved in decision-making.")
        suggestions.append("Maintain high levels of participatory care.")

    # Clarity of care plan and communication
    if instance_data.get('Exam_Explained', 3) < 3:
        reasons.append("Medical exams were not clearly explained.")
        suggestions.append("Improve communication around procedures and clinical steps.")

    if instance_data.get('Discuss_NextSteps', 3) < 3:
        reasons.append("Next steps in the care journey were not well communicated.")
        suggestions.append("Ensure every client knows what to expect after each visit.")

    # Structural/Contextual
    if instance_data.get('Employment_Grouped') in ['Unemployed', 'Unknown']:
        reasons.append("Client's unemployment status may affect care experience or stress levels.")
        suggestions.append("Offer counseling and support services for unemployed clients.")

    if instance_data.get('Education_Grouped') in ['None', 'Primary']:
        reasons.append("Lower education level may be linked with reduced care understanding.")
        suggestions.append("Simplify communication and use visual aids for clarity.")

    if instance_data.get('Facility_Care_Dur_Years', 0) < 1:
        reasons.append("Short duration of care at this facility may limit relationship-building.")
        suggestions.append("Strengthen early rapport and onboarding for new clients.")

    if instance_data.get('HIV_Care_Duration_Ratio', 0.0) < 0.3:
        reasons.append("Low proportion of time spent in care may affect satisfaction.")
        suggestions.append("Reinforce retention efforts and build long-term trust.")

    return len(reasons) > 0, reasons, suggestions

RULES = [
    (
        'Empathy and listening were key factors.',
        "Encourage strong provider-client communication and emotional intelligence.",
        (lambda d: d.get('Empathy_Listening_Interaction', 15) < 9 or d.get('Empathy_Listening_Interaction', 15) > 15, True)
    ),
    (
        'Decision-sharing and empathy influenced satisfaction.',
        "Promote client-centered decision-making practices.",
        (lambda d: d.get('Empathy_DecisionShare_Interaction', 15) < 9 or d.get('Empathy_DecisionShare_Interaction', 15) > 15, True)
    ),
    (
        'Exam clarity and next-step planning mattered.',
        "Make sure clients understand their exams and what comes next.",
        (lambda d: d.get('Exam_Explained', 3) < 3 or d.get('Discuss_NextSteps', 3) < 3, True)
    ),
    (
        'Overall Client Satisfaction influenced by multiple clinical and contextual factors.',
        "Address communication, education, employment, treatment duration, and participatory care.",
        (rule_overall_client_satisfaction, True)
    )
]

def deepseek_generate_explanation(prediction, confidence, top_features, reasons, suggestions, openrouter_api_key):
    """Generates a detailed explanation using the GenAI model."""
    if not openrouter_api_key:
        return "GenAI explanation unavailable: API key not configured."

    prompt = f"""
    You are an AI assistant helping a healthcare team understand why a specific HIV client was predicted to be '{prediction}' with {confidence} confidence.

    Top contributing factors (from SHAP analysis):
    {json.dumps(top_features, indent=2)}

    Key issues/influences identified by our rule-based system:
    {reasons}

    Suggestions for improvement or reinforcement:
    {suggestions}

    Based on all the information above, please synthesize a concise and actionable explanation for the clinical quality improvement team. Focus on what this prediction means in a real-world context and what practical steps can be taken.
    """
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistralai/mistral-7b-instruct:free", # Using a free model for demonstration
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, data=json.dumps(body), timeout=30) # Increased timeout
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logger.error(f"GenAI API request failed: {e}")
        return f"Error connecting to GenAI service: {e}. Check API key and network."
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected GenAI API response format: {e}. Response: {response.text if 'response' in locals() else 'No response'}")
        return "Error parsing GenAI service response."
    except Exception as e:
        logger.error(f"An unexpected error occurred in GenAI explanation: {e}")
        return f"An unexpected error occurred during GenAI explanation: {e}"

# Main Explanation Pipeline (adapted for direct use in Streamlit)
def explain_catboost_prediction_integrated(model, X_test, background_data, categorical_cols, openrouter_api_key):
    """
    Generates prediction and explanation for a single instance.
    This is an integrated version of the original explain_catboost_prediction.
    """
    # Ensure X_test is a DataFrame with correct dtypes
    X_test = enforce_categorical_dtypes(X_test.copy(), categorical_cols)
    
    # Ensure X_test has only one row for single instance prediction
    if X_test.shape[0] != 1:
        logger.error(f"Expected single instance (1 row) for X_test, but got {X_test.shape[0]} rows.")
        raise ValueError("X_test must contain exactly one instance for explanation.")
    
    instance = X_test.iloc[0:1] # Still use slicing to maintain DataFrame structure for SHAP

    explainer = get_shap_explainer(model)

    # Predict raw values and probabilities
    preds_proba = model.predict_proba(instance)[0] # [0] to get probabilities for the single instance
    pred_class = np.argmax(preds_proba)
    confidence_val = round(float(np.max(preds_proba)) * 100, 1)
    confidence = f"{confidence_val}%"

    # SHAP values
    shap_values_raw = explainer.shap_values(instance)
    
    # Determine the SHAP values for the predicted class
    if isinstance(shap_values_raw, list):
        # Multi-class: list of arrays, each array is (n_instances, n_features)
        # Convert the list of arrays to a single 3D numpy array, then select the correct slice
        # This is the key change to robustly handle multi-class SHAP outputs
        shap_values_array = np.array(shap_values_raw) # Shape will be (n_classes, 1, n_features)
        shap_vals_for_class = shap_values_array[pred_class, 0, :] # Select class, then instance, then all features
        shap_base_value = explainer.expected_value[pred_class]
    else:
        # Binary: single array of (n_instances, n_features)
        shap_vals_for_class = shap_values_raw[0] # Get the first (and only) instance's SHAP values
        shap_base_value = explainer.expected_value

    # Ensure it's truly 1D for consistency before passing to .tolist()
    # Use .flatten() to guarantee 1D array
    shap_vals_for_class = np.array(shap_vals_for_class).flatten()
    if shap_vals_for_class.ndim != 1:
        logger.error(f"SHAP values for class are not 1D after flatten: {shap_vals_for_class.shape}")
        raise ValueError("SHAP values for single instance are not 1D after processing.")

    shap_dict = dict(zip(X_test.columns, shap_vals_for_class.flatten()))
    top_features = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
    top_features = {k: round(float(v), 3) for k, v in top_features.items()}

    instance_data = instance.iloc[0].to_dict()

    reasons, suggestions = [], []
    for reason_text, suggestion_text, rule_tuple in RULES:
        rule_fn, expects_instance_data = rule_tuple
        if expects_instance_data:
            result = rule_fn(instance_data)
            if isinstance(result, tuple):
                is_triggered, rule_reasons, rule_suggestions = result
                if is_triggered:
                    reasons.extend(rule_reasons)
                    suggestions.extend(rule_suggestions)
            else: # For rules that return just a boolean
                if result:
                    reasons.append(reason_text)
                    suggestions.append(suggestion_text)

    mapped_pred = label_map.get(int(pred_class), "Unknown")

    explanation_text = deepseek_generate_explanation(
        mapped_pred, confidence, top_features, reasons, suggestions, openrouter_api_key=openrouter_api_key
    )

    return {
        'prediction': mapped_pred,
        'confidence': confidence,
        'top_features': top_features,
        'reason': "; ".join(reasons) if reasons else "No specific rule-based issues detected.",
        'suggestions': "; ".join(suggestions) if suggestions else "Continue standard best practices.",
        'genai_explanation': explanation_text,
        'shap_values': shap_vals_for_class.tolist(),
        'shap_base_value': shap_base_value # Include base value for plotting
    }

# --- Main Streamlit App Logic ---
# Only proceed if all artifacts were loaded successfully
if model is not None and top_features is not None and categories is not None and label_encoder is not None:
    st.subheader("Client & Provider Interaction Details")
    
    with st.form("predict_form"):
        raw_inputs = {}
        st.markdown("#### Patient Demographics")
        demo_cols = st.columns(3)
        raw_inputs['Age'] = demo_cols[0].number_input("Age", min_value=0, value=35)
        raw_inputs['Employment_Grouped'] = demo_cols[1].selectbox("Employment Status", categories.get('Employment_Grouped', ['Employed', 'Unemployed']))
        raw_inputs['State'] = demo_cols[2].selectbox("State", categories.get('State', ['State A', 'State B']))
        # Add 'Education_Grouped' as it's used in rule_overall_client_satisfaction and is a categorical feature
        raw_inputs['Education_Grouped'] = demo_cols[0].selectbox("Education Level", categories.get('Education_Grouped', ['None', 'Primary', 'Secondary', 'Tertiary']))


        st.markdown("#### Care Duration (in Years)")
        care_cols = st.columns(3)
        raw_inputs['HIV_Duration_Years'] = care_cols[0].number_input("Duration of HIV Diagnosis (Years)", min_value=0.0, value=5.0, format="%.1f")
        raw_inputs['Care_Duration_Years'] = care_cols[1].number_input("Duration at Current Facility (Years)", min_value=0.0, value=2.0, format="%.1f")
        raw_inputs['Facility_Care_Dur_Years'] = care_cols[2].number_input("Total Duration of Care (All Facilities, Years)", min_value=0.0, value=5.0, format="%.1f")

        st.markdown("#### Provider Interaction Scores (Rate 1-5)")
        interaction_cols = st.columns(3)
        raw_inputs['Empathy_Score'] = interaction_cols[0].slider("Average Empathy Score", 1.0, 5.0, 4.0)
        raw_inputs['Listening_Score'] = interaction_cols[1].slider("Average Listening Score", 1.0, 5.0, 4.0)
        raw_inputs['Decision_Share_Score'] = interaction_cols[2].slider("Average Decision Sharing Score", 1.0, 5.0, 3.0)
        
        st.markdown("#### Communication & Information")
        comm_cols = st.columns(2)
        raw_inputs['Exam_Explained'] = comm_cols[0].selectbox("The provider explained exams/procedures clearly.", options=likert_options, index=3)
        raw_inputs['Discuss_NextSteps'] = comm_cols[1].selectbox("The provider discussed the next steps in my care.", options=likert_options, index=3)

        submitted = st.form_submit_button("Predict Client Satisfaction")

    if submitted:
        # --- On-the-fly Feature Engineering ---
        final_features = {}
        
        # Calculate HIV Care Duration Ratio
        final_features['HIV_Care_Duration_Ratio'] = raw_inputs['HIV_Duration_Years'] / (raw_inputs['Care_Duration_Years'] + 0.1)
        
        # Calculate Interaction Features
        final_features['Empathy_Listening_Interaction'] = raw_inputs['Empathy_Score'] * raw_inputs['Listening_Score']
        final_features['Empathy_DecisionShare_Interaction'] = raw_inputs['Empathy_Score'] * raw_inputs['Decision_Share_Score']
        
        # Map Likert scale text to numeric values
        final_features['Exam_Explained'] = likert_map[raw_inputs['Exam_Explained']]
        final_features['Discuss_NextSteps'] = likert_map[raw_inputs['Discuss_NextSteps']]

        # Create a DataFrame with all expected top_features, initialized to 0 or appropriate default
        # This ensures the DataFrame has all columns the model expects, even if not directly from UI
        input_df = pd.DataFrame(columns=top_features)
        
        # Populate with calculated and direct raw inputs
        for feature, value in final_features.items():
            if feature in input_df.columns:
                input_df.loc[0, feature] = value
        
        # Add remaining direct raw inputs that might not have been processed into final_features yet
        for feature in ['Age', 'Employment_Grouped', 'State', 'Facility_Care_Dur_Years', 'Education_Grouped']:
            if feature in raw_inputs and feature in input_df.columns:
                input_df.loc[0, feature] = raw_inputs[feature]
            elif feature in input_df.columns and feature not in raw_inputs:
                # This case should ideally be handled by UI inputs or defaults
                # If a feature is in top_features but not in raw_inputs, it needs a default.
                # For categorical: use first category. For numeric: use 0.
                if feature in categories:
                    input_df.loc[0, feature] = categories[feature][0]
                else:
                    input_df.loc[0, feature] = 0.0 # Default numeric to 0.0

        # Fill any remaining NaN values (for features in top_features not covered by inputs)
        # This ensures the DataFrame is complete before passing to the model
        for col in input_df.columns:
            if pd.isna(input_df.loc[0, col]):
                if col in categories:
                    input_df.loc[0, col] = categories[col][0]
                else:
                    input_df.loc[0, col] = 0.0 # Default numeric to 0.0

        # Ensure categorical columns have 'category' dtype for CatBoost
        categorical_features_in_model = [col for col in top_features if col in categories]
        input_df = enforce_categorical_dtypes(input_df, categorical_features_in_model)

        # Display features sent to the model for debugging/transparency
        st.write("### Features Sent to Model")
        st.json({k: (f"{v:.2f}" if isinstance(v, float) else v) for k,v in input_df.iloc[0].to_dict().items()})

        # --- Direct Prediction and Explanation Logic ---
        # Get the API Key for GenAI
        openrouter_api_key = os.getenv("SATISFACTION_APP_KEY")
        if not openrouter_api_key:
            st.warning("⚠️ GenAI API key (SATISFACTION_APP_KEY) not found. GenAI explanations will be unavailable.")
            logger.warning("SATISFACTION_APP_KEY not found in environment.")

        try:
            # Directly call the explanation function
            result = explain_catboost_prediction_integrated(
                model=model,
                X_test=input_df,
                background_data=pd.DataFrame(np.zeros((1, len(top_features))), columns=top_features), # Dummy background data
                categorical_cols=categorical_features_in_model,
                openrouter_api_key=openrouter_api_key
            )

            st.success(f"**Prediction:** {result['prediction']} (Confidence {result['confidence']})")
            
            st.subheader("Top 3 Contributing Factors (SHAP)")
            st.json(result.get("top_features", {}))

            if result.get("suggestions"):
                st.info(f"**AI Suggestions:** {result['suggestions']}")

            st.subheader("Full GenAI Explanation")
            st.markdown(result.get("genai_explanation", "No explanation provided."))

            st.subheader("SHAP Visualization")
            try:
                # Use np.array().flatten() to ensure 1D arrays
                shap_vals = np.array(result.get("shap_values", [])).flatten()
                shap_base_value = result.get("shap_base_value", 0) 
                
                if shap_vals.size == 0:
                    st.warning("No SHAP values returned for visualization.")
                else:
                    feature_values_for_shap_plot = np.array(input_df.iloc[0].values).flatten()
                    feature_names_for_shap_plot = list(input_df.columns)

                    # Validate lengths before creating Explanation object
                    if not (len(shap_vals) == len(feature_names_for_shap_plot) == len(feature_values_for_shap_plot)):
                        st.error(f"Dimension mismatch for SHAP plotting: SHAP values ({len(shap_vals)}), Feature Names ({len(feature_names_for_shap_plot)}), Feature Data ({len(feature_values_for_shap_plot)}).")
                        raise ValueError("Dimension mismatch for SHAP plotting.")

                    # Create a single Explanation object
                    single_explanation = shap.Explanation(
                        values=shap_vals,
                        base_values=shap_base_value,
                        data=feature_values_for_shap_plot,
                        feature_names=feature_names_for_shap_plot
                    )
                    # Pass the single Explanation object directly to waterfall plot
                    st_shap(shap.plots.waterfall(single_explanation), height=400)

            except Exception as e:
                st.warning(f"Could not generate SHAP plot: {e}")
                logger.error(f"Error generating SHAP plot: {e}")

        except Exception as e:
            st.error(f"An error occurred during prediction or explanation: {e}")
            logger.exception("Error during prediction or explanation.")
else:
    st.warning("Application artifacts (model, features, categories, encoder) could not be loaded. Please check the 'model/' directory and ensure files exist.")

st.divider()
# Removed the logs section as it relied on the FastAPI /logs endpoint
# For a single Streamlit app, you'd typically manage logs within Streamlit's session state
# if you wanted to display a history of predictions.
st.info("Prediction history is not stored in this single-app deployment.")