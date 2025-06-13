# src/ann_train.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.pytorch

from src.data import get_merged_data
from src.load_colab_split import load_fixed_colab_split
from src.preprocessing import preprocess_and_split_clean as preprocess_and_split
from src.ann_model import CreditANN
from config import (
    RANDOM_STATE, MODEL_PATH, ANN_MODEL_FILE, BATCH_SIZE,
    ANN_HIDDEN1, ANN_HIDDEN2, ANN_DROPOUT, ANN_LR, ANN_EPOCHS
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_PATH, exist_ok=True)

# ✅ Load top-20 feature indices
TOP_20_INDICES_PATH = os.path.join(MODEL_PATH, "top20_feature_indices.pkl")
top_20_indices = joblib.load(TOP_20_INDICES_PATH)

class CreditDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_ann():
    mlflow.set_experiment("credit_risk_classification")
    with mlflow.start_run(run_name="ANN"):


        X_train, X_test, y_train, y_test = load_fixed_colab_split()

        # ✅ Convert sparse to dense
        X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
        X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    
       

        # ✅ Create loaders
        train_loader = DataLoader(CreditDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(CreditDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

        input_dim = X_train.shape[1]
        model = CreditANN(input_dim).to(device)

        pos_weight_value = (y_train.shape[0] - y_train.sum()) / y_train.sum()
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=ANN_LR)

        model.train()
        for epoch in range(ANN_EPOCHS):
            running_loss = 0.0
            correct, total = 0, 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{ANN_EPOCHS}")

            for batch_X, batch_y in loop:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                running_loss += loss.item()

                loop.set_postfix(loss=loss.item(), acc=correct / total)

        # --- Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        all_preds = [p[0] for p in all_preds]
        all_labels = [l[0] for l in all_labels]

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)

        print("\n🤖 ANN Evaluation on Test Set:")
        print(f"Accuracy:   {acc:.4f}")
        print(f"F1 Score:   {f1:.4f}")
        print(f"ROC AUC:    {auc:.4f}")

        # --- MLflow logging
        mlflow.log_param("model", "ANN")
        mlflow.log_param("hidden1", ANN_HIDDEN1)
        mlflow.log_param("hidden2", ANN_HIDDEN2)
        mlflow.log_param("dropout", ANN_DROPOUT)
        mlflow.log_param("learning_rate", ANN_LR)
        mlflow.log_param("epochs", ANN_EPOCHS)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        torch.save(model.state_dict(), ANN_MODEL_FILE)
        mlflow.log_artifact(ANN_MODEL_FILE)
        print(f"\n✅ ANN model saved to {ANN_MODEL_FILE}")

if __name__ == "__main__":
    train_ann()
