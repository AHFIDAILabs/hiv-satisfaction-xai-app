# niceGUI_app/Home.py

from nicegui import ui
import pandas as pd
import joblib
import os
from pathlib import Path
from app.model_utils import get_model
from app.explanation_engine import explain_catboost_prediction

# Load model and sample data
model = get_model()
data_path = Path('data/processed_data.csv')
df = pd.read_csv(data_path)
background_df = df.sample(n=100, random_state=42)

# Load only the top 10 features used in the final model
feature_path = Path('model/important_features.joblib')
top_features = joblib.load(feature_path)
categorical_cols = ['Education_Grouped', 'State', 'Employment_Grouped']

# Instructions
ui.label("HIV Client Satisfaction Prediction + Explanation").classes("text-2xl font-bold")
ui.label("Upload a single instance CSV for prediction and explanation")

upload_result = ui.label("Waiting for upload...")

def handle_upload(e):
    upload_result.text = f"Processing: {e.name}"
    try:
        instance_df = pd.read_csv(e.content)
        missing = set(top_features) - set(instance_df.columns)
        for col in missing:
            instance_df[col] = pd.NA
        instance_df = instance_df[top_features]

        log = explain_catboost_prediction(
            instance_idx=0,
            model=model,
            X_test=instance_df,
            background_data=background_df[top_features],
            categorical_cols=categorical_cols,
            openrouter_api_key=os.getenv("SATISFACTION_APP_KEY")
        )

        ui.label(f"Prediction: {log['prediction']} ({log['confidence']})").classes("text-xl text-green-700")
        ui.label(f"Top features: {log['top_features']}").classes("text-md")
        ui.label(f"Explanation: {log['genai_explanation']}").classes("text-md mt-4")

    except Exception as ex:
        ui.label(f"❌ Error: {ex}").classes("text-red-600")

ui.upload(on_upload=handle_upload, label="📤 Upload Instance CSV")

ui.run(title="HIV Client Satisfaction Explainer Dashboard")