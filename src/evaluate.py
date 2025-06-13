# src/evaluate.py

import joblib
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from config import RF_MODEL_FILE, XGB_MODEL_FILE, ANN_MODEL_FILE, BATCH_SIZE
from src.data import get_merged_data
from src.preprocessing import preprocess_and_split
from src.ann_train import CreditANN, CreditDataset, device

def evaluate_model(name, model, X_test, y_test, is_ann=False):
    if is_ann:
        # Convert to dense tensors
        X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test
        dataset = CreditDataset(X_test, y_test)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        model.eval()
        all_preds, all_labels, all_proba = [], [], []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
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

    print(f"\n📊 {name} Evaluation:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC AUC:    {auc:.4f}")
    return acc, f1, auc

def main():
    df = get_merged_data()
    X_train, X_test, y_train, y_test, _ = preprocess_and_split(df)

    # Load models
    rf = joblib.load(RF_MODEL_FILE)
    xgb = joblib.load(XGB_MODEL_FILE)

    input_dim = X_test.shape[1] if not hasattr(X_test, "toarray") else X_test.toarray().shape[1]
    ann = CreditANN(input_dim).to(device)
    ann.load_state_dict(torch.load(ANN_MODEL_FILE, map_location=device))

    print("\n🔍 Comparing Trained Models:")
    evaluate_model("Random Forest", rf, X_test, y_test)
    evaluate_model("XGBoost", xgb, X_test, y_test)
    evaluate_model("ANN", ann, X_test, y_test, is_ann=True)

if __name__ == "__main__":
    main()
