# src/preprocessing.py
print("✅ preprocessing.py loaded")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from credit_risk_classifier.config import TEST_SIZE, RANDOM_STATE, DOWNSAMPLE_RATIO
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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test, preprocessor

# 🚀 NEW FUNCTION FOR FASTAPI USER INPUT
def preprocess_user_input(request):
    """
    Converts user input dict into model's expected 20 feature values (as list)
    """
    features = [
        (request.loan_amount + request.max_past_loan) / 2,       # AMT_CREDIT_SUM_mean
        request.loan_amount + request.max_past_loan,            # AMT_CREDIT_SUM_sum
        request.loan_amount * 0.1,                              # INST_AMT_PAYMENT_MAX
        request.loan_amount * 0.05,                             # INST_AMT_PAYMENT_MEAN
        request.loan_amount * 0.6,                              # AMT_CREDIT_SUM_DEBT_max
        -12 * request.loan_term_years / 2,                      # POS_MONTHS_BALANCE_MEAN
        0.5,                                                    # INST_PAYMENT_RATIO_MIN
        request.past_loans,                                      # SK_ID_BUREAU_sum
        request.loan_amount * 0.1,                              # INST_AMT_INSTALMENT_MAX
        request.loan_amount * request.loan_term_years * 12,     # INST_AMT_INSTALMENT_SUM
        request.loan_amount * 0.5,                              # AMT_CREDIT_SUM_DEBT_mean
        -12 * request.loan_term_years,                          # POS_MONTHS_BALANCE_MIN
        request.loan_amount * 0.5,                              # AMT_CREDIT_SUM_DEBT_sum
        1.0 if request.higher_education else 0.0,               # NAME_EDUCATION_TYPE_Higher education
        request.loan_amount * request.loan_term_years * 12 * 0.05, # INST_AMT_PAYMENT_SUM
        -12 * request.loan_term_years,                          # POS_MONTHS_BALANCE_SUM
        -365 * (request.years_since_first_loan + request.years_since_last_loan), # DAYS_CREDIT_sum
        -365 * request.years_since_first_loan,                  # DAYS_CREDIT_min
        -365 * request.years_since_last_loan,                   # DAYS_CREDIT_max
        -365 * (request.years_since_first_loan + request.years_since_last_loan) / 2 # DAYS_CREDIT_mean
    ]
    return features
