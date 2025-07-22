# app/model_utils.py

import joblib
import pandas as pd

MODEL_PATH = "model/top10_model.joblib"
ENCODER_PATH = "model/label_encoder.joblib"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def get_model():
    return model

def get_encoder():
    return encoder

def predict(data: pd.DataFrame):
    categorical = data.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical:
        data[categorical] = encoder.transform(data[categorical])
    preds = model.predict(data)
    probs = model.predict_proba(data)
    return preds, probs