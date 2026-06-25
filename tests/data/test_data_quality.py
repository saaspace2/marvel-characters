import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope='module')
def train_data():
    spark = SparkSession.builder.getOrCreate()
    return spark.table('mlops_dev.marvel_characters.train_set').toPandas()

def test_no_duplicate_ids(train_data):
    assert train_data['Id'].nunique() == len(train_data)

def test_required_columns_exist(train_data):
    required = ['Height','Weight','Universe','Identity','Gender',
                'Marital_Status','Teams','Origin','Magic','Mutant','Alive']
    missing = [c for c in required if c not in train_data.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"

def test_no_nulls_in_numeric_features(train_data):
    for col in ['Height', 'Weight']:
        assert train_data[col].isnull().sum() == 0

def test_target_is_binary(train_data):
    unique_vals = set(train_data['Alive'].dropna().unique())
    assert unique_vals.issubset({0, 1})

def test_no_overlap_between_train_and_test(train_data, test_data):
    overlap = set(train_data['Id']) & set(test_data['Id'])
    assert len(overlap) == 0

def test_class_balance_acceptable(train_data):
    alive_pct = train_data['Alive'].mean()
    assert 0.10 <= alive_pct <= 0.90
