from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
import numpy as np
from credit_risk_classifier.ann_model import CreditANN
from credit_risk_classifier.config import (
    XGB_MODEL_FILE, RF_MODEL_FILE, ANN_MODEL_FILE
)

app = FastAPI(title="Credit Risk Classifier API")

# Load models
try:
    xgb_model = joblib.load(XGB_MODEL_FILE)
    rf_model = joblib.load(RF_MODEL_FILE)
    ann_model = CreditANN(xgb_model.n_features_in_)
    ann_model.load_state_dict(torch.load(ANN_MODEL_FILE, map_location="cpu"))
    ann_model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

num_features = xgb_model.n_features_in_

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Credit Risk Classifier API is live."}

@app.post("/predict/xgb")
def predict_xgb(request: PredictRequest):
    if len(request.features) != num_features:
        raise HTTPException(status_code=400, detail=f"Expected {num_features} features.")
    
    X = np.array(request.features).reshape(1, -1)
    proba = xgb_model.predict_proba(X)[0, 1]
    pred = int(proba > 0.5)
    return {
        "prediction": pred,
        "probability": round(float(proba), 4)
    }

@app.post("/predict/ensemble")
def predict_ensemble(request: PredictRequest):
    if len(request.features) != num_features:
        raise HTTPException(status_code=400, detail=f"Expected {num_features} features.")
    
    X = np.array(request.features).reshape(1, -1)
    rf_p = rf_model.predict_proba(X)[0, 1]
    xgb_p = xgb_model.predict_proba(X)[0, 1]
    with torch.no_grad():
        ann_p = torch.sigmoid(ann_model(torch.tensor(X, dtype=torch.float32))).numpy()[0, 0]
    final_p = 0.2 * rf_p + 0.6 * xgb_p + 0.2 * ann_p
    pred = int(final_p > 0.5)
    return {
        "prediction": pred,
        "probability": round(float(final_p), 4)
    }
