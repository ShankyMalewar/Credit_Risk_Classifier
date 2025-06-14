# src/predict.py

import joblib
import torch
import numpy as np
import pandas as pd
import os

from src.ann_model import CreditANN
from config import RF_MODEL_FILE, XGB_MODEL_FILE, ANN_MODEL_FILE, PREPROCESSOR_FILE, MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load top-20 feature indices
TOP_20_INDICES_PATH = os.path.join(MODEL_PATH, "top20_feature_indices.pkl")
top_20_indices = joblib.load(TOP_20_INDICES_PATH)

# ---------------- Loaders ---------------- #
def load_rf():
    return joblib.load(RF_MODEL_FILE)

def load_xgb():
    return joblib.load(XGB_MODEL_FILE)

def load_ann(input_dim):
    model = CreditANN(input_dim).to(device)
    model.load_state_dict(torch.load(ANN_MODEL_FILE, map_location=device))
    model.eval()
    return model

def load_preprocessor():
    return joblib.load(PREPROCESSOR_FILE)

# ---------------- Prediction Wrapper ---------------- #
def predict(model_type="rf", raw_input_dict=None):
    if raw_input_dict is None:
        raise ValueError("Provide input data as a dictionary")

    df_input = pd.DataFrame([raw_input_dict])
    preprocessor = load_preprocessor()

    # Transform input
    X_transformed = preprocessor.transform(df_input)
    X_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed

    # ✅ Slice top 20 features
    X_top20 = X_dense[:, top_20_indices]

    if model_type == "rf":
        model = load_rf()
        proba = model.predict_proba(X_top20)[:, 1]
        pred = (proba > 0.5).astype(int)

    elif model_type == "xgb":
        model = load_xgb()
        proba = model.predict_proba(X_top20)[:, 1]
        pred = (proba > 0.5).astype(int)

    elif model_type == "ann":
        model = load_ann(X_top20.shape[1])
        with torch.no_grad():
            inputs = torch.tensor(X_top20, dtype=torch.float32).to(device)
            output = torch.sigmoid(model(inputs)).cpu().numpy().flatten()
            proba = output
            pred = (proba > 0.5).astype(int)

    else:
        raise ValueError("Model type must be 'rf', 'xgb', or 'ann'.")

    return {"prediction": int(pred[0]), "probability": float(proba[0])}

# ---------------- Example ---------------- #
if __name__ == "__main__":
    sample_input = {
        "AMT_CREDIT": 200000,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        # ✅ This should match the raw feature format from application_train.csv + merged features
        # If testing manually, ensure full set of required fields is present
    }

    result = predict(model_type="rf", raw_input_dict=sample_input)
    print("✅ Prediction:", result)
