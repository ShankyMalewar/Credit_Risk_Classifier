# src/ensemble.py

import os
import joblib
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.data import get_merged_data
from src.preprocessing import preprocess_and_split_clean as preprocess_and_split
from src.ann_model import CreditANN
from config import RF_MODEL_FILE, XGB_MODEL_FILE, ANN_MODEL_FILE, MODEL_PATH
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ann_model(input_dim, device):
    model = CreditANN(input_dim).to(device)
    model.load_state_dict(torch.load(ANN_MODEL_FILE, map_location=device))
    model.eval()
    return model

def evaluate_ensemble(rf_pred, xgb_pred, ann_pred, y_test):
    final_proba = (rf_pred + xgb_pred + ann_pred) / 3
    final_pred = (final_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, final_pred)
    f1 = f1_score(y_test, final_pred)
    auc = roc_auc_score(y_test, final_proba)

    print("\n🧠 Ensemble Evaluation:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC AUC:    {auc:.4f}")

    return acc, f1, auc

def main():
    mlflow.set_experiment("credit_risk_classification")
    with mlflow.start_run(run_name="Ensemble"):
        df = get_merged_data()
        X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)

        X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

        rf = joblib.load(RF_MODEL_FILE)
        xgb = joblib.load(XGB_MODEL_FILE)
        ann = load_ann_model(X_test_dense.shape[1], device)

        rf_pred = rf.predict_proba(X_test)[:, 1]
        xgb_pred = xgb.predict_proba(X_test)[:, 1]

        with torch.no_grad():
            ann_input = torch.tensor(X_test_dense, dtype=torch.float32).to(device)
            ann_output = torch.sigmoid(ann(ann_input)).cpu().numpy()
            ann_pred = ann_output[:, 0]

        acc, f1, auc = evaluate_ensemble(rf_pred, xgb_pred, ann_pred, y_test)

        # Log parameters
        mlflow.log_param("model", "Ensemble")
        mlflow.log_param("ensemble_type", "Average")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log a dummy artifact to represent ensemble logic
        os.makedirs(MODEL_PATH, exist_ok=True)
        dummy_path = os.path.join(MODEL_PATH, "ensemble_logic.txt")
        with open(dummy_path, "w") as f:
            f.write("This run represents an average ensemble of RF, XGB, and ANN models.\n")
            f.write("Weights: 1/3 each\n")
            f.write("Voting strategy: Probability average -> threshold > 0.5")


        mlflow.log_artifact(dummy_path)
        print(f"✅ Ensemble logic artifact logged.")

if __name__ == "__main__":
    main()
