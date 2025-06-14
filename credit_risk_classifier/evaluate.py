import os
import argparse
import joblib
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from credit_risk_classifier.config import RF_MODEL_FILE, XGB_MODEL_FILE, ANN_MODEL_FILE, BATCH_SIZE, MODEL_PATH
from credit_risk_classifier.load_colab_split import load_fixed_colab_split
from credit_risk_classifier.ann_model import CreditANN
from credit_risk_classifier.logger import get_logger

logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_20_INDICES_PATH = os.path.join(MODEL_PATH, "top20_feature_indices.pkl")
top_20_indices = joblib.load(TOP_20_INDICES_PATH)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=str, choices=["rf", "xgb", "ann", "ensemble", "all"],
                        required=True, help="Model to evaluate")
    return parser.parse_args()

def evaluate_model(name, model, X_test, y_test, is_ann=False):
    X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    if is_ann:
        dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model.eval()
        all_preds, all_labels, all_proba = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_proba.extend(probs.cpu().numpy())
        all_preds = [p[0] for p in all_preds]
        all_labels = [l[0] for l in all_labels]
        all_proba = [p[0] for p in all_proba]
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        all_preds = y_pred
        all_proba = y_proba
        all_labels = y_test.values

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_proba)

    logger.info(f"{name} Evaluation:")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"F1 Score:   {f1:.4f}")
    logger.info(f"ROC AUC:    {auc:.4f}")
    return acc, f1, auc

def evaluate_ensemble(X_test, y_test):
    rf = joblib.load(RF_MODEL_FILE)
    xgb = joblib.load(XGB_MODEL_FILE)
    ann = CreditANN(len(top_20_indices)).to(device)
    ann.load_state_dict(torch.load(ANN_MODEL_FILE, map_location=device))
    ann.eval()

    X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    rf_pred = rf.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict_proba(X_test)[:, 1]
    with torch.no_grad():
        ann_out = torch.sigmoid(ann(torch.tensor(X_test, dtype=torch.float32).to(device))).cpu().numpy()[:, 0]

    final_proba = (0.3 * rf_pred) + (0.5 * xgb_pred) + (0.2 * ann_out)
    final_pred = (final_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, final_pred)
    f1 = f1_score(y_test, final_pred)
    auc = roc_auc_score(y_test, final_proba)

    logger.info("Ensemble Evaluation:")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"F1 Score:   {f1:.4f}")
    logger.info(f"ROC AUC:    {auc:.4f}")
    return acc, f1, auc

def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_fixed_colab_split()

    if args.model in ["rf", "all"]:
        rf = joblib.load(RF_MODEL_FILE)
        evaluate_model("Random Forest", rf, X_test, y_test)

    if args.model in ["xgb", "all"]:
        xgb = joblib.load(XGB_MODEL_FILE)
        evaluate_model("XGBoost", xgb, X_test, y_test)

    if args.model in ["ann", "all"]:
        ann = CreditANN(len(top_20_indices)).to(device)
        ann.load_state_dict(torch.load(ANN_MODEL_FILE, map_location=device))
        evaluate_model("ANN", ann, X_test, y_test, is_ann=True)

    if args.model in ["ensemble", "all"]:
        evaluate_ensemble(X_test, y_test)

if __name__ == "__main__":
    main()
