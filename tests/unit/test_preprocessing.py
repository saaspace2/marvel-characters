import pytest, pandas as pd, numpy as np
from your_package.preprocessing import clean_data, encode_features

def test_clean_data_removes_nulls():
    dirty = pd.DataFrame({'Height': [1.75, None], 'Weight': [70.0, None]})
    result = clean_data(dirty)
    assert result.isnull().sum().sum() == 0

def test_clean_data_keeps_required_columns():
    df = pd.DataFrame({'Height':[1.75], 'Weight':[70.0]})
    result = clean_data(df)
    assert 'Height' in result.columns

def test_encode_features_produces_numbers():
    df = pd.DataFrame({'Gender': ['Male', 'Female', 'Male']})
    result = encode_features(df)
    assert result['Gender'].dtype in [np.int64, np.float64]
