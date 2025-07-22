# ==========================================================
# ✅ app/explanation_engine.py: SHAP + EXPLANATION PIPELINE
# ==========================================================
# This module provides functions to explain model predictions using SHAP and a GenAI model.
# It includes functions to generate explanations and apply rule-based reasoning for satisfaction scoring.
# ==========================

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

# --------------------------------------------
# ✅ Load API key from .env
# --------------------------------------------
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

openrouter_api_key = os.getenv("SATISFACTION_APP_KEY")
if not openrouter_api_key:
    raise EnvironmentError("❌ SATISFACTION_APP_KEY not found in .env file or environment")

label_map = {
    0: 'Very Dissatisfied',
    1: 'Dissatisfied',
    2: 'Neutral',
    3: 'Satisfied',
    4: 'Very Satisfied'
}

logs_df = pd.DataFrame(columns=[
    'instance_idx', 'prediction', 'confidence', 'top_features',
    'reason', 'suggestions', 'genai_explanation'
])

def rule_overall_client_satisfaction(shap_scores, instance_data):
    reasons, suggestions = [] , []
    if shap_scores.get('Empathy_Score', 3) >= 3.5:
        reasons.append("High Empathy Score contributing to satisfaction.")
        suggestions.append("Maintain strong empathetic communication.")
    if shap_scores.get('Listening_Score', 3) >= 3.5:
        reasons.append("High Listening Score indicating good provider listening.")
        suggestions.append("Continue active listening techniques.")
    if shap_scores.get('Decision_Share_Score', 3) < 2.5:
        reasons.append("Lack of participatory care (low decision sharing) negatively impacted satisfaction.")
        suggestions.append("Improve patient engagement in healthcare decisions.")
    if shap_scores.get('Empathy_Score', 3) < 2.5:
        reasons.append("Poor provider attitude (low empathy) negatively impacted satisfaction.")
        suggestions.append("Enhance provider's empathetic communication and attitude training.")
    if shap_scores.get('Listening_Score', 3) < 2.5:
        reasons.append("Poor provider listening negatively impacted satisfaction.")
        suggestions.append("Train providers on active listening techniques.")
    if instance_data.get('Family_Setting') == 'Polygamous':
        reasons.append("Client from a polygamous family setting associated with lower satisfaction.")
        suggestions.append("Provide targeted support for clients from polygamous families.")
    return len(reasons) > 0, reasons, suggestions

RULES = [
    ('Empathy was a key factor', "Focus on cultivating supportive and warm provider relationships.", (lambda s: s.get('Empathy_Score', 3) < 2.5 or s.get('Empathy_Score', 3) >= 3.5, False)),
    ('Decision-sharing was a key factor', "Improve client involvement in healthcare decisions.", (lambda s: s.get('Decision_Share_Score', 3) < 2.5, False)),
    ('Listening was a key factor', "Enhance provider active listening techniques.", (lambda s: s.get('Listening_Score', 3) < 2.5 or s.get('Listening_Score', 3) >= 3.5, False)),
    ('Overall Client Satisfaction influenced by multiple factors',
     "Address a combination of provider communication, service delivery, socio-economic factors, treatment complexity, participatory care, family dynamics, treatment duration, and structural/financial barriers.",
     (rule_overall_client_satisfaction, True))
]

def enforce_categorical_dtypes(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df

def get_shap_explainer(model, background_data):
    return shap.TreeExplainer(model)

def deepseek_generate_explanation(prediction, confidence, top_features, reasons, suggestions, openrouter_api_key):
    prompt = f"""
    You are an AI assistant helping a healthcare team understand why a specific HIV client was predicted to be '{prediction}' with {confidence}% confidence.

    Top contributing factors:
    {json.dumps(top_features, indent=2)}

    Rule-based issues/influences:
    {reasons}

    Suggestions for improvement/reinforcement:
    {suggestions}

    Based on the article summary \"Journey to Sustainability...\" write a concise and actionable explanation for clinical quality improvement.
    """
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "moonshotai/kimi-k2:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                 headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "LLM error: " + response.text
    except Exception as e:
        return f"Exception: {e}"

def explain_catboost_prediction(instance_idx, model, X_test, background_data, categorical_cols, openrouter_api_key):
    global logs_df

    X_test = enforce_categorical_dtypes(X_test.copy(), categorical_cols)
    background_data = enforce_categorical_dtypes(background_data.copy(), categorical_cols)
    explainer = get_shap_explainer(model, background_data)
    instance = X_test.iloc[instance_idx:instance_idx+1]
    shap_vals = explainer.shap_values(instance)
    preds = model.predict_proba(instance)[0]
    pred_class = model.predict(instance)

    confidence_val = round(float(np.max(preds)) * 100, 1)
    confidence = f"{confidence_val}%"

    shap_vals_row = shap_vals[np.argmax(preds)][0] if isinstance(shap_vals, list) else shap_vals[0]
    shap_dict = dict(zip(X_test.columns, shap_vals_row.flatten()))
    top_features = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
    top_features = {k: round(float(v), 1) for k, v in top_features.items()}

    shap_scores = {
        'Empathy_Score': instance['Empathy_Score'].iloc[0],
        'Decision_Share_Score': instance['Decision_Share_Score'].iloc[0],
        'Listening_Score': instance['Listening_Score'].iloc[0],
    }

    instance_data = instance.iloc[0].to_dict()

    reasons, suggestions = [], []
    for reason_text, suggestion_text, rule_tuple in RULES:
        rule_fn, expects_instance_data = rule_tuple
        if expects_instance_data:
            is_triggered, rule_reasons, rule_suggestions = rule_fn(shap_scores, instance_data)
            if is_triggered:
                reasons.extend(rule_reasons)
                suggestions.extend(rule_suggestions)
        else:
            if rule_fn(shap_scores):
                reasons.append(reason_text)
                suggestions.append(suggestion_text)

    mapped_pred = label_map.get(int(pred_class), str(pred_class))

    explanation_text = deepseek_generate_explanation(
        mapped_pred, confidence, top_features, reasons, suggestions, openrouter_api_key=openrouter_api_key
    )

    log_entry = {
        'instance_idx': instance_idx,
        'prediction': mapped_pred,
        'confidence': confidence,
        'top_features': top_features,
        'reason': "; ".join(reasons),
        'suggestions': "; ".join(suggestions),
        'genai_explanation': explanation_text
    }

    logs_df = pd.concat([logs_df, pd.DataFrame([log_entry])], ignore_index=True)
    return log_entry