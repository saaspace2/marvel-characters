import pytest, mlflow
from mlflow import MlflowClient

MODEL_NAME = 'mlops_dev.marvel_characters.marvel_character_model_custom'

def test_model_exists_in_registry(client):
    models = client.search_registered_models(filter_string=f"name='{MODEL_NAME}'")
    assert len(models) > 0

def test_latest_model_alias_exists(client):
    version = client.get_model_version_by_alias(name=MODEL_NAME, alias='latest-model')
    assert version is not None

def test_model_has_required_tags(client):
    version = client.get_model_version_by_alias(name=MODEL_NAME, alias='latest-model')
    assert 'git_sha' in version.tags
    assert 'branch' in version.tags

def test_model_has_signature(client):
    model_info = mlflow.models.get_model_info(f'models:/{MODEL_NAME}@latest-model')
    assert model_info.signature is not None
