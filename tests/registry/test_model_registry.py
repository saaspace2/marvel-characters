"""Model Registry Tests for BOTH Marvel Characters models in Unity Catalog."""

import pandas as pd
import pytest
import mlflow
from mlflow.types import DataType

MODEL_NAMES = {
    "basic": "mlops_dev.marvel_characters.marvel_character_model_basic",
    "custom": "mlops_dev.marvel_characters.marvel_character_model_custom",
}


@pytest.fixture(params=list(MODEL_NAMES.values()), ids=list(MODEL_NAMES.keys()))
def model_name(request):
    return request.param


def _build_sample_input(signature) -> pd.DataFrame:
    """Build a dummy row matching the signature's exact declared types.

    Uses the actual MLflow DataType enum for comparison instead of parsing
    the string representation of col.type (which includes a 'DataType.'
    prefix and therefore never matched 'long' or 'double' directly, causing
    every column -- including numeric ones -- to receive the string 'Unknown',
    which MLflow's schema enforcement then rejected with a type mismatch error).

    Type mapping:
      DataType.double / DataType.float   ->  0.0  (float)
      DataType.long   / DataType.integer ->  0    (int)
      everything else (string, binary)   ->  "Unknown"
    """
    row = {}
    for col in signature.inputs.inputs:
        if col.type in (DataType.double, DataType.float):
            row[col.name] = [0.0]
        elif col.type in (DataType.long, DataType.integer):
            row[col.name] = [0]
        else:
            row[col.name] = ["Unknown"]
    return pd.DataFrame(row)


# ---------------------------------------------------------------------------
# Registry existence & metadata tests
# ---------------------------------------------------------------------------

def test_model_exists_in_registry(client, model_name):
    """Model must be present in Unity Catalog under the expected full name."""
    try:
        model = client.get_registered_model(name=model_name)
    except Exception as e:
        pytest.fail(f"{model_name} not found in registry: {e}")
    assert model is not None


def test_model_has_at_least_one_version(client, model_name):
    """Model must have at least one registered version -- empty registry = broken pipeline."""
    versions = list(client.search_model_versions(filter_string=f"name='{model_name}'"))
    assert len(versions) > 0, "Model exists but has no registered versions"


def test_latest_model_alias_exists(client, model_name):
    """'latest-model' alias must exist so the serving endpoint can locate the model."""
    version = client.get_model_version_by_alias(name=model_name, alias="latest-model")
    assert version is not None


def test_latest_alias_points_to_highest_version(client, model_name):
    """'latest-model' alias must point to the most recently registered version.

    If the alias lags behind the highest version number it means the training
    pipeline registered a new version but forgot to update the alias, so the
    serving endpoint would silently keep using an older model.
    """
    aliased_version = client.get_model_version_by_alias(name=model_name, alias="latest-model")
    all_versions = list(client.search_model_versions(filter_string=f"name='{model_name}'"))
    highest_version_number = max(int(v.version) for v in all_versions)
    assert int(aliased_version.version) == highest_version_number, (
        f"'latest-model' alias points to version {aliased_version.version}, "
        f"but the highest registered version is {highest_version_number}"
    )


def test_model_has_required_tags(client, model_name):
    """git_sha and branch tags must be present and non-empty for traceability.

    Without these tags there is no way to link a registered model back to the
    exact source commit that produced it, making debugging impossible.
    """
    version = client.get_model_version_by_alias(name=model_name, alias="latest-model")
    assert "git_sha" in version.tags, "Missing required tag: git_sha"
    assert "branch" in version.tags, "Missing required tag: branch"
    assert version.tags["git_sha"], "Tag 'git_sha' is present but empty"
    assert version.tags["branch"], "Tag 'branch' is present but empty"


def test_model_version_has_valid_run_id(client, model_name):
    """The registered version must reference a real, readable MLflow run.

    A missing or dangling run_id means the training metrics and artifacts
    logged alongside the model cannot be retrieved.
    """
    version = client.get_model_version_by_alias(name=model_name, alias="latest-model")
    assert version.run_id, "Model version has no run_id"
    run = client.get_run(version.run_id)
    assert run is not None, f"run_id {version.run_id} does not resolve to a valid run"


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------

def test_model_has_signature(model_name):
    """Model must have a saved input/output signature.

    Without a signature MLflow cannot enforce schema validation at inference
    time, meaning type mismatches silently produce wrong predictions instead
    of raising a clear error.
    """
    model_info = mlflow.models.get_model_info(f"models:/{model_name}@latest-model")
    assert model_info.signature is not None, "Model has no signature"
    assert len(model_info.signature.inputs.inputs) > 0, "Model signature has no input columns"


# ---------------------------------------------------------------------------
# Load-and-predict test
# ---------------------------------------------------------------------------

def test_model_can_be_loaded_and_predicts(model_name):
    """Model must load without error and produce valid predictions on a dummy row.

    For the basic model:  predictions must be binary (0 or 1).
    For the custom model: the result dict / DataFrame must contain the
                          'Survival prediction' key produced by adjust_predictions().

    Root cause of the previous failure
    -----------------------------------
    _build_sample_input() was comparing str(col.type).lower() to plain strings
    like 'long' and 'double'.  MLflow's DataType.__str__ returns 'DataType.long'
    and 'DataType.double' (with the class prefix), so those equality checks
    always fell through to the else branch, filling every column -- including
    numeric Height / Weight / Teams / Magic / Mutant -- with the string
    "Unknown".  MLflow's schema enforcement then raised:

        MlflowException: Failed to convert column Height
                         from type object to DataType.double.

    The fix is to compare col.type directly against the DataType enum members,
    which is both correct and immune to any future changes in __str__ formatting.
    """
    model_uri = f"models:/{model_name}@latest-model"
    model_info = mlflow.models.get_model_info(model_uri)
    sample_input = _build_sample_input(model_info.signature)

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    result = loaded_model.predict(sample_input)

    assert result is not None, "Model returned None instead of predictions"

    if model_name.endswith("_custom"):
        # Custom model wraps output in adjust_predictions() -> {"Survival prediction": [...]}
        if isinstance(result, dict):
            assert "Survival prediction" in result, (
                f"Custom model output dict missing 'Survival prediction' key. Got: {result}"
            )
        else:
            assert "Survival prediction" in getattr(result, "columns", []), (
                f"Custom model output DataFrame missing 'Survival prediction' column. "
                f"Got columns: {list(getattr(result, 'columns', []))}"
            )
    else:
        # Basic model -> binary integer predictions (0 or 1)
        values = result.values.tolist() if hasattr(result, "values") else list(result)
        assert len(values) == len(sample_input), (
            f"Expected {len(sample_input)} predictions, got {len(values)}"
        )
        assert all(v in (0, 1) for v in values), (
            f"Expected binary predictions (0 or 1), got: {values}"
        )
