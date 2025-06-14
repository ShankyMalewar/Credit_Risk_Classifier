import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from credit_risk_classifier.config import (
    CV_SPLITS, RANDOM_STATE, MODEL_PATH,
    RF_MODEL_FILE, XGB_MODEL_FILE, ANN_MODEL_FILE,
    BATCH_SIZE, ANN_LR, ANN_EPOCHS
)
from credit_risk_classifier.ann_model import CreditANN
from credit_risk_classifier.load_colab_split import load_fixed_colab_split
from credit_risk_classifier.logger import get_logger

logger = get_logger(__name__, log_file="ml_pipeline.log")
logger.info("Logger initialized. Starting training pipeline...")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_PATH, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train credit risk classifier")
    parser.add_argument("--model", type=str, choices=["rf", "xgb", "ann", "ensemble", "all"],
                        required=True, help="Model to train: rf, xgb, ann, ensemble, or all")
    return parser.parse_args()

def log_artifact():
    mlflow.log_artifact("ml_pipeline.log")
    logger.info("Logged ml_pipeline.log as artifact.")

def train_rf(X_train, y_train):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    for train_idx, val_idx in skf.split(X_train, y_train):
        rf.fit(X_train[train_idx], y_train.iloc[train_idx])
    return rf

def train_xgb(X_train, y_train):
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_ann(X_train, y_train):
    train_data = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CreditANN(X_train.shape[1]).to(device)
    pos_weight = (y_train.shape[0] - y_train.sum()) / y_train.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=ANN_LR)

    model.train()
    for epoch in range(ANN_EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), ANN_MODEL_FILE)
    return model

def evaluate_model(model, X_test, y_test, label="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"{label} Evaluation:")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"F1 Score:   {f1:.4f}")
    logger.info(f"ROC AUC:    {auc:.4f}")
    return acc, f1, auc

def evaluate_ann(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32).to(device))).cpu().numpy()
    preds = (out > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, out)
    logger.info("ANN Evaluation:")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"F1 Score:   {f1:.4f}")
    logger.info(f"ROC AUC:    {auc:.4f}")
    return acc, f1, auc

def evaluate_ensemble(X_test, y_test):
    rf = joblib.load(RF_MODEL_FILE)
    xgb = joblib.load(XGB_MODEL_FILE)
    ann = CreditANN(X_test.shape[1]).to(device)
    ann.load_state_dict(torch.load(ANN_MODEL_FILE, map_location=device))
    ann.eval()

    rf_pred = rf.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict_proba(X_test)[:, 1]
    with torch.no_grad():
        ann_out = torch.sigmoid(torch.tensor(X_test, dtype=torch.float32).to(device))
        ann_pred = ann(ann_out).cpu().numpy()[:, 0]

    final_proba = 0.3 * rf_pred + 0.5 * xgb_pred + 0.2 * ann_pred
    final_pred = (final_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, final_pred)
    f1 = f1_score(y_test, final_pred)
    auc = roc_auc_score(y_test, final_proba)

    logger.info("Ensemble Evaluation:")
    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"F1 Score:   {f1:.4f}")
    logger.info(f"ROC AUC:    {auc:.4f}")

def main():
    args = parse_args()
    X_train, X_test, y_train, y_test = load_fixed_colab_split()

    mlflow.set_experiment("credit_risk_classification")

    if args.model in ["rf", "all"]:
        logger.info("Starting Random Forest training...")
        with mlflow.start_run(run_name="RandomForest"):
            rf_model = train_rf(X_train, y_train)
            logger.info("Random Forest training completed.")
            evaluate_model(rf_model, X_test, y_test, label="Random Forest")
            joblib.dump(rf_model, RF_MODEL_FILE)
            logger.info(f"Random Forest model saved to {RF_MODEL_FILE}")
            log_artifact()

    if args.model in ["xgb", "all"]:
        logger.info("Starting XGBoost training...")
        with mlflow.start_run(run_name="XGBoost"):
            xgb_model = train_xgb(X_train, y_train)
            logger.info("XGBoost training completed.")
            evaluate_model(xgb_model, X_test, y_test, label="XGBoost")
            joblib.dump(xgb_model, XGB_MODEL_FILE)
            logger.info(f"XGBoost model saved to {XGB_MODEL_FILE}")
            log_artifact()

    if args.model in ["ann", "all"]:
        logger.info("Starting ANN training...")
        with mlflow.start_run(run_name="ANN"):
            ann_model = train_ann(X_train, y_train)
            logger.info("ANN training completed.")
            evaluate_ann(ann_model, X_test, y_test)
            logger.info(f"ANN model saved to {ANN_MODEL_FILE}")
            log_artifact()

    if args.model == "ensemble":
        logger.info("Evaluating ensemble...")
        with mlflow.start_run(run_name="Ensemble"):
            evaluate_ensemble(X_test, y_test)
            log_artifact()

if __name__ == "__main__":
    main()
