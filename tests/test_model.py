import joblib
import torch
import numpy as np
from credit_risk_classifier.config import RF_MODEL_FILE, XGB_MODEL_FILE, ANN_MODEL_FILE
from credit_risk_classifier.ann_model import CreditANN

def test_rf_model_loads():
    rf = joblib.load(RF_MODEL_FILE)
    assert rf is not None

def test_xgb_model_loads():
    xgb = joblib.load(XGB_MODEL_FILE)
    assert xgb is not None

def test_ann_model_loads_and_predicts():
    
    input_dim = 20  
    model = CreditANN(input_dim)
    model.load_state_dict(torch.load(ANN_MODEL_FILE, map_location="cpu"))
    model.eval()

    X_dummy = torch.zeros((1, input_dim), dtype=torch.float32)
    with torch.no_grad():
        output = torch.sigmoid(model(X_dummy)).numpy()

    assert output.shape == (1, 1)
    assert 0.0 <= output[0, 0] <= 1.0

def test_xgb_predict_shape_and_range():
    xgb = joblib.load(XGB_MODEL_FILE)
    X_dummy = np.zeros((1, xgb.n_features_in_))
    proba = xgb.predict_proba(X_dummy)[0, 1]
    assert 0.0 <= proba <= 1.0
