# app/explanation_engine.py

import shap
import pandas as pd
import numpy as np
import json
import requests
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

openrouter_api_key = os.getenv("SATISFACTION_APP_KEY")
if not openrouter_api_key:
    logging.warning("SATISFACTION_APP_KEY not found in .env file. Ensure it's set in the environment.")

label_map = {
    0: 'Very Dissatisfied',
    1: 'Dissatisfied',
    2: 'Neutral',
    3: 'Satisfied',
    4: 'Very Satisfied'
}

# --------------------------------------------
# ✅ [UPDATED] Rule-Based Logic
# --------------------------------------------
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

# --------------------------------------------
# ✅ Helper Functions
# --------------------------------------------
def enforce_categorical_dtypes(df, categorical_cols):
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def get_shap_explainer(model):
    return shap.TreeExplainer(model)

def deepseek_generate_explanation(prediction, confidence, top_features, reasons, suggestions, openrouter_api_key):
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
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, data=json.dumps(body), timeout=20)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"GenAI API request failed: {e}")
        return f"Error connecting to GenAI service: {e}"
    except (KeyError, IndexError) as e:
        logging.error(f"Unexpected GenAI API response format: {e}")
        return "Error parsing GenAI service response."

# --------------------------------------------
# ✅ Main Explanation Pipeline (Note: This function will be called directly from Home.py)
# --------------------------------------------
def explain_catboost_prediction(instance_idx, model, X_test, background_data, categorical_cols, openrouter_api_key):
    # This function's logic is now integrated into explain_catboost_prediction_integrated in Home.py
    # This file is kept for modularity, but the actual call will be to the integrated version.
    # If you intend to use this exact function, ensure it's imported and called correctly.
    # For this single-app scenario, the logic is duplicated/adapted in Home.py for direct use.
    # If you want to strictly keep this function and import it, then Home.py would call this.
    # For simplicity of single-app, I've moved the core logic to Home.py's
    # explain_catboost_prediction_integrated.
    pass # This function is effectively replaced by explain_catboost_prediction_integrated in Home.py