# HIV-Satisfaction-XAI-APP
Interactive app predicting and explaining HIV client satisfaction with patient-centered care using CatBoost &amp; SHAP). Built with FastAPI &amp; NiceGUI, deployable on Hugging Face Spaces.


# 🧠 CatBoost Model Explanation Dashboard

This is a fully functional FastAPI + NiceGUI application for training, predicting, and explaining machine learning models (CatBoost-based) using SHAP. Deploy-ready on Hugging Face Spaces.

## 🔧 Features
- Upload and preprocess your data
- Train a CatBoost model with cross-validation and hyperparameter tuning
- Visualize top 10 features
- Get predictions with confidence scores
- Generate SHAP-based explanations

## 📁 Project Structure
```bash
.
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── app/
│   ├── api.py
│   ├── explanation_engine.py
│   ├── model_utils.py
│   └── log_cache.json
├── niceGUI_app/
│   └── Home.py
├── train_model.py
├── data_preprocessing.py
├── main.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started (Local)
```bash
pip install -r requirements.txt
python data_preprocessing.py
python train_model.py
python main.py
```
Then open another terminal and run:
```bash
python niceGUI_app/Home.py
```

## 🧩 Deployment on Hugging Face Spaces
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create a new Space → Select **Python** template
3. Upload all files
4. Ensure you have a `data/raw/data.csv` in the repo
5. Hugging Face will run `app.py` or `main.py` as the default entry point

### Space Configuration (`README.md`, `requirements.txt`, and `main.py`) is already set.

**Important**: Hugging Face Spaces doesn’t support simultaneous FastAPI + browser UIs directly. You may need to:
- Wrap everything under NiceGUI (including API endpoints)
- Or deploy API and GUI separately (Render + HF combo)

---

## 👨‍⚕️ Built by a doctor-data scientist hybrid
Good at treating both patients and pipelines.

For questions or issues, please open an [issue](https://huggingface.co/spaces) or reach out.

