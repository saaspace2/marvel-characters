import pytest, mlflow
from mlflow import MlflowClient

@pytest.fixture(scope='module')
def trained_model():
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        name='mlops_dev.marvel_characters.marvel_character_model_basic',
        alias='latest-model'
    )
    return mlflow.sklearn.load_model(f'models:/{version.model_id}')

def test_accuracy_above_threshold(trained_model, test_data):
    X, y = test_data
    accuracy = trained_model.score(X, y)
    assert accuracy > 0.70, f"Accuracy {accuracy:.3f} below 0.70"

def test_model_not_overfitting(trained_model, test_data):
    # ... load train data ...
    gap = train_acc - test_acc
    assert gap < 0.10, f"Overfitting: gap={gap:.3f}"

def test_predictions_are_binary(trained_model, test_data):
    X, y = test_data
    preds = trained_model.predict(X)
    assert set(preds).issubset({0, 1})
