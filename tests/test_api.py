from fastapi.testclient import TestClient
from credit_risk_classifier.api import app  

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk Classifier API is live."}

def test_xgb_predict_valid():
    features = [0.0] * 20  
    response = client.post("/predict/xgb", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data

def test_xgb_predict_invalid():
   
    features = [0.0] * 5
    response = client.post("/predict/xgb", json={"features": features})
    assert response.status_code == 422 or response.status_code == 400

def test_ensemble_predict_valid():
    features = [0.0] * 20 
    response = client.post("/predict/ensemble", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data

def test_ensemble_predict_invalid():
    features = [0.0] * 5 
    response = client.post("/predict/ensemble", json={"features": features})
    assert response.status_code in [400, 422]

def test_xgb_predict_missing_body():
    response = client.post("/predict/xgb", json={})  
    assert response.status_code in [400, 422]

def test_ensemble_predict_missing_body():
    response = client.post("/predict/ensemble", json={})
    assert response.status_code in [400, 422]
