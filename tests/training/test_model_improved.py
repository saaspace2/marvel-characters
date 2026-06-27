"""Tests for BasicModel.model_improved() -- the method that decides whether
a newly trained model should replace the currently registered one.

This is mocked rather than hitting real MLflow, since we need full control
over both branches: no baseline existing yet, vs. comparing against a real
baseline with a known F1 score. This is the most consequential piece of
decision logic in the whole pipeline, and previously had zero test coverage.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from marvel_characters.config import Tags
from marvel_characters.models.basic_model import BasicModel


def _make_model():
    config = SimpleNamespace(
        num_features=["Height", "Weight"],
        cat_features=["Gender"],
        target="Alive",
        parameters={"n_estimators": 5, "verbose": -1},
        catalog_name="test_catalog",
        schema_name="test_schema",
        experiment_name_basic="/exp/test",
    )
    tags = Tags(git_sha="abc", branch="main")
    model = BasicModel(config, tags, spark=None)
    model.eval_data = None  # not used directly since mlflow.models.evaluate is fully mocked
    return model


@patch("marvel_characters.models.basic_model.MlflowClient")
def test_model_improved_returns_true_when_no_baseline_exists(mock_client_cls):
    """First-ever run in an environment: nothing to compare against, so the
    new model should be treated as an improvement by default."""
    model = _make_model()
    model.metrics = {"f1_score": 0.80}

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = Exception("not found")
    mock_client_cls.return_value = mock_client

    assert model.model_improved() is True


@patch("marvel_characters.models.basic_model.mlflow")
@patch("marvel_characters.models.basic_model.MlflowClient")
def test_model_improved_returns_true_when_new_f1_is_higher(mock_client_cls, mock_mlflow):
    model = _make_model()
    model.metrics = {"f1_score": 0.85}

    mock_client = MagicMock()
    mock_version = MagicMock()
    mock_version.model_id = "abc123"
    mock_client.get_model_version_by_alias.return_value = mock_version
    mock_client_cls.return_value = mock_client

    mock_mlflow.models.evaluate.return_value.metrics = {"f1_score": 0.70}

    assert model.model_improved() is True


@patch("marvel_characters.models.basic_model.mlflow")
@patch("marvel_characters.models.basic_model.MlflowClient")
def test_model_improved_returns_false_when_new_f1_is_lower(mock_client_cls, mock_mlflow):
    model = _make_model()
    model.metrics = {"f1_score": 0.60}

    mock_client = MagicMock()
    mock_version = MagicMock()
    mock_version.model_id = "abc123"
    mock_client.get_model_version_by_alias.return_value = mock_version
    mock_client_cls.return_value = mock_client

    mock_mlflow.models.evaluate.return_value.metrics = {"f1_score": 0.85}

    assert model.model_improved() is False
