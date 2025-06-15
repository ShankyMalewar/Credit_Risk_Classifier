import pandas as pd
from credit_risk_classifier.data import get_merged_data

def test_data_loads():
    df = get_merged_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_data_columns_expected():
    df = get_merged_data()
    expected_cols = ["TARGET"]  
    for col in expected_cols:
        assert col in df.columns

def test_no_null_target():
    df = get_merged_data()
    assert df["TARGET"].isnull().sum() == 0
