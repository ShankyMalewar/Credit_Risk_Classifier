# src/preprocessing.py
print("✅ preprocessing.py loaded")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.config import TEST_SIZE, RANDOM_STATE, DOWNSAMPLE_RATIO
import joblib
import os

def split_target_features(df):
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    return X, y

def downsample(X, y):
    df = X.copy()
    df['TARGET'] = y

    major = df[df['TARGET'] == 0]
    minor = df[df['TARGET'] == 1]

    major_downsampled = major.sample(n=len(minor) * DOWNSAMPLE_RATIO, random_state=RANDOM_STATE)
    df_balanced = pd.concat([major_downsampled, minor], axis=0).sample(frac=1, random_state=RANDOM_STATE)

    X_bal = df_balanced.drop(columns=["TARGET"])
    y_bal = df_balanced["TARGET"]
    return X_bal, y_bal

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Convert all categorical to strings (safety)
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor, categorical_cols, numerical_cols

def preprocess_and_split_clean(df, save_path="models/preprocessor.pkl"):
    X, y = split_target_features(df)
    X_bal, y_bal = downsample(X, y)

    preprocessor, cat_cols, num_cols = build_preprocessor(X_bal)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=TEST_SIZE, stratify=y_bal, random_state=RANDOM_STATE
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Save fitted preprocessor
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test, preprocessor
