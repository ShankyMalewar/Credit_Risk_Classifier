import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import xgboost as xgb

from src.data import get_merged_data
from src.load_colab_split import load_fixed_colab_split
from src.preprocessing import preprocess_and_split_clean as preprocess_and_split
from config import (
    CV_SPLITS, RANDOM_STATE, MODEL_PATH,
    RF_MODEL_FILE, XGB_MODEL_FILE, PREPROCESSOR_FILE,
    REGISTER_RF_NAME, REGISTER_XGB_NAME
)

os.makedirs(MODEL_PATH, exist_ok=True)

# ✅ Load Top 20 Feature Indices
TOP_20_INDICES_PATH = os.path.join(MODEL_PATH, "top20_feature_indices.pkl")
top_20_indices = joblib.load(TOP_20_INDICES_PATH)

def train_rf(X_train, y_train):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_val)

        f1 = f1_score(y_val, preds)
        f1_scores.append(f1)
        print(f"🌲 Fold {fold + 1} RF F1: {f1:.4f}")

    print(f"\n✅ RF Mean F1 (CV): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    return rf

def train_xgb(X_train, y_train):
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test, label="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📊 Evaluation on Test Set ({label}):")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC AUC:    {auc:.4f}")

    return acc, f1, auc

def main():
    mlflow.set_experiment("credit_risk_classification")

  
    X_train, X_test, y_train, y_test= load_fixed_colab_split()

    # ✅ Convert to dense if sparse
    X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

  

    # --- Random Forest Logging --- #
    with mlflow.start_run(run_name="RandomForest"):
        print("\n🌲 Training Random Forest...")
        rf_model = train_rf(X_train, y_train)
        acc, f1, auc = evaluate_model(rf_model, X_test, y_test, label="Random Forest")

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 12)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        joblib.dump(rf_model, RF_MODEL_FILE)
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="rf_model",
            registered_model_name=REGISTER_RF_NAME,
            signature=infer_signature(X_test, rf_model.predict(X_test)),
            input_example=X_test[:2]
        )
        mlflow.log_artifact(PREPROCESSOR_FILE)
        print(f"✅ RF model saved to {RF_MODEL_FILE}")

    # --- XGBoost Logging --- #
    with mlflow.start_run(run_name="XGBoost"):
        print("\n🚀 Training XGBoost...")
        xgb_model = train_xgb(X_train, y_train)
        acc, f1, auc = evaluate_model(xgb_model, X_test, y_test, label="XGBoost")

        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        joblib.dump(xgb_model, XGB_MODEL_FILE)
        mlflow.sklearn.log_model(
            sk_model=xgb_model,
            artifact_path="xgb_model",
            registered_model_name=REGISTER_XGB_NAME,
            signature=infer_signature(X_test, xgb_model.predict(X_test)),
            input_example=X_test[:2]
        )
        mlflow.log_artifact(PREPROCESSOR_FILE)
        print(f"✅ XGB model saved to {XGB_MODEL_FILE}")

if __name__ == "__main__":
    main()
