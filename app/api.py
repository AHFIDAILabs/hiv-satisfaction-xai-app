# app/api.py 

import logging
import joblib
import pandas as pd
import numpy as np
import sys
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv

from app.explanation_engine import explain_catboost_prediction

# ------------------------
# Logging & Environment Setup
# ------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("api")

# ------------------------
# FastAPI App Setup
# ------------------------
app = FastAPI(title="HIV Client Satisfaction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------
# Load Model Artifacts
# ------------------------
try:
    model = joblib.load("model/top10_model.joblib")
    top_features = joblib.load("model/important_features.joblib")
    # ✅ Added API Key loading
    openrouter_api_key = os.getenv("SATISFACTION_APP_KEY")
    if not openrouter_api_key:
        raise EnvironmentError("SATISFACTION_APP_KEY not found")
    
    # ✅ Added placeholder for SHAP background data
    background_data = pd.DataFrame(np.ones((1, len(top_features))), columns=top_features)

    logger.info(f"Loaded model and top features: {top_features}")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise RuntimeError(f"Could not load model artifacts: {e}")

# ------------------------
# Logs Storage
# ------------------------
logs = []

# ------------------------
# Input Schema
# ------------------------
class FeatureInput(BaseModel):
    features: Dict[str, Any]

# ------------------------
# Root Endpoint
# ------------------------
@app.get("/")
def root():
    logger.info("Root endpoint accessed")
    return {"message": "API running"}

# ------------------------
# Prediction Endpoint
# ------------------------
@app.post("/predict")
def predict(input_data: FeatureInput, request: Request):
    logger.info(f"Prediction request payload: {input_data.features}")

    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_data.features])

        # Ensure all top features are present
        missing_cols = set(top_features) - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan  # Add missing features with NaN

        # Reorder columns to match model training
        df = df[top_features]

        # ✅ Corrected function call with proper arguments
        result = explain_catboost_prediction(
            instance_idx=len(logs),
            model=model,
            X_test=df,
            background_data=background_data,
            categorical_cols=[], # Assuming categorical columns are handled if necessary
            openrouter_api_key=openrouter_api_key
        )
        logs.append(result)

        logger.info(f"Prediction result: {result}")
        return result

    except KeyError as e:
        logger.error(f"Missing required feature: {e}")
        raise HTTPException(status_code=400, detail=f"Missing required feature: {e}")
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# ------------------------
# Logs Endpoint
# ------------------------
@app.get("/logs")
def get_logs():
    return logs