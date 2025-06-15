# src/data.py

import os
import pandas as pd
import numpy as np
from credit_risk_classifier.config import DATA_PATH

def downcast_df(df):
    for col in df.select_dtypes(include='float64').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include='int64').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def load_application_data():
    app = pd.read_csv(os.path.join(DATA_PATH, "application_train.csv"))
    app = downcast_df(app)
    return app

def process_bureau():
    bureau = downcast_df(pd.read_csv(os.path.join(DATA_PATH, "bureau.csv")))

    leak_cols = [
        'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT',
        'AMT_CREDIT_MAX_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY'
    ]
    bureau.drop(columns=leak_cols, inplace=True, errors='ignore')
    bureau.drop(columns=['CREDIT_CURRENCY'], inplace=True, errors='ignore')
    bureau = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE', 'CREDIT_TYPE'], drop_first=True)
    bureau = bureau.select_dtypes(include=[np.number])

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(['mean', 'max', 'min', 'sum'])
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.reset_index(inplace=True)
    return bureau_agg

def process_previous():
    prev = downcast_df(pd.read_csv(os.path.join(DATA_PATH, "previous_application.csv")))

    leaky_cols = [
        'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
        'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'DAYS_DECISION', 'AMT_DOWN_PAYMENT'
    ]
    prev.drop(columns=leaky_cols, inplace=True, errors='ignore')

    cat_cols = ['NAME_CONTRACT_TYPE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']
    prev[cat_cols] = prev[cat_cols].astype(str)
    prev_cat = pd.get_dummies(prev[cat_cols], drop_first=True)
    prev_encoded = pd.concat([prev[['SK_ID_CURR']], prev_cat], axis=1)
    prev_encoded = prev_encoded.groupby('SK_ID_CURR').mean().reset_index()

    num_agg = prev.groupby('SK_ID_CURR').agg({
        'AMT_APPLICATION': ['mean', 'max', 'min', 'sum'],
        'AMT_CREDIT': ['mean', 'max', 'min', 'sum'],
        'CNT_PAYMENT': ['mean', 'max', 'min'],
        'NFLAG_INSURED_ON_APPROVAL': ['sum', 'mean']
    })
    num_agg.columns = ['_'.join(col).strip() for col in num_agg.columns]
    num_agg.reset_index(inplace=True)

    return pd.merge(num_agg, prev_encoded, on='SK_ID_CURR', how='left')

def process_installments():
    df = downcast_df(pd.read_csv(os.path.join(DATA_PATH, "installments_payments.csv")))
    df.drop(columns=['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT'], inplace=True, errors='ignore')
    df['PAYMENT_RATIO'] = df['AMT_PAYMENT'] / (df['AMT_INSTALMENT'] + 1e-6)

    agg = df.groupby('SK_ID_CURR').agg({
        'AMT_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT': ['mean', 'max', 'sum'],
        'PAYMENT_RATIO': ['mean', 'max', 'min'],
        'SK_ID_PREV': 'nunique',
        'NUM_INSTALMENT_NUMBER': 'max'
    })
    agg.columns = ['INST_' + '_'.join(col).upper() for col in agg.columns]
    agg.reset_index(inplace=True)
    return agg

def process_credit_card():
    df = downcast_df(pd.read_csv(os.path.join(DATA_PATH, "credit_card_balance.csv")))
    df.drop(columns=['SK_ID_PREV'], inplace=True, errors='ignore')
    df = df.select_dtypes(include=['number'])

    agg = df.groupby('SK_ID_CURR').agg(['mean', 'max', 'min', 'sum'])
    agg.columns = ['CC_' + '_'.join(col).upper() for col in agg.columns]
    agg.reset_index(inplace=True)
    return agg

def process_pos_cash():
    df = downcast_df(pd.read_csv(os.path.join(DATA_PATH, "POS_CASH_balance.csv")))
    df.drop(columns=['SK_ID_PREV'], inplace=True, errors='ignore')
    df = df.select_dtypes(include=['number'])

    agg = df.groupby('SK_ID_CURR').agg(['mean', 'max', 'min', 'sum'])
    agg.columns = ['POS_' + '_'.join(col).upper() for col in agg.columns]
    agg.reset_index(inplace=True)
    return agg

def get_merged_data():
    df = load_application_data()

    # Merge all sources
    df = df.merge(process_bureau(), on='SK_ID_CURR', how='left')
    df = df.merge(process_previous(), on='SK_ID_CURR', how='left')
    df = df.merge(process_installments(), on='SK_ID_CURR', how='left')
    df = df.merge(process_credit_card(), on='SK_ID_CURR', how='left')
    df = df.merge(process_pos_cash(), on='SK_ID_CURR', how='left')

    df.fillna(0, inplace=True)
    df = df.drop(columns=['SK_ID_CURR'])
    return df

