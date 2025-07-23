# app/model_utils.py

import joblib
import pandas as pd

MODEL_PATH = "model/top10_model.joblib"
ENCODER_PATH = "model/label_encoder.joblib"

# These are loaded once when the module is imported
# In Streamlit, it's better to load them with @st.cache_resource
# as done in Home.py to avoid reloading on every rerun.
# So, these functions might not be strictly necessary if Home.py loads directly.
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def get_model():
    return model

def get_encoder():
    return encoder

def predict(data: pd.DataFrame):
    # This predict function assumes the data is already preprocessed and encoded if necessary.
    # In the single-app scenario, preprocessing and encoding happen in Home.py before prediction.
    categorical = data.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical:
        # This part might need adjustment depending on how your label_encoder was trained
        # and if CatBoost expects raw or encoded categoricals.
        # For CatBoost, often it's better to pass raw categoricals and let CatBoost handle them
        # if the model was trained with raw categoricals.
        data_copy = data.copy()
        for col in categorical:
            if col in encoder.feature_names_in_: # Check if encoder knows this feature
                # Ensure categories match encoder's categories to avoid errors
                # This is a simplification; robust handling of unseen categories is complex.
                known_categories = list(encoder.categories_[list(encoder.feature_names_in_).index(col)])
                data_copy[col] = pd.Categorical(data_copy[col], categories=known_categories)
                data_copy[col] = encoder.transform(data_copy[col])
            else:
                # Handle cases where a categorical column is not in the encoder's known features
                # e.g., convert to NaN or a default category if appropriate
                data_copy[col] = np.nan # Or some other default
        data = data_copy

    preds = model.predict(data)
    probs = model.predict_proba(data)
    return preds, probs