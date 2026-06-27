"""Shared fixtures for Model Training Tests.

Unlike Data Quality Tests (checking the data) or unit/integration tests
(checking code logic), these check whether the TRAINED MODEL ITSELF is
actually good: does it beat a naive guess, is it overfitting, are its raw
outputs valid. Read-only: nothing is written or modified.

Precondition: a model must already exist in Unity Catalog with the
'latest-model' alias set (i.e. your training/deployment job has run at
least once). If it hasn't, trained_model() will fail with a "not found"
error from MLflow -- that's expected, not a bug in these tests.
"""

import mlflow
import pytest
from databricks.connect import DatabricksSession
from mlflow import MlflowClient
from types import SimpleNamespace

CATALOG_NAME = "mlops_dev"
SCHEMA_NAME = "marvel_characters"
MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.marvel_character_model_basic"

# Matches the profile name shown by `databricks auth profiles`.
DATABRICKS_PROFILE = "dbc-34ada41e-0aa1"

NUM_FEATURES = ["Height", "Weight"]
CAT_FEATURES = ["Universe", "Identity", "Gender", "Marital_Status", "Teams", "Origin", "Magic", "Mutant"]
TARGET = "Alive"


@pytest.fixture(scope="module", autouse=True)
def _configure_mlflow():
    """Point MLflow at the same workspace/profile used everywhere else."""
    mlflow.set_tracking_uri(f"databricks://{DATABRICKS_PROFILE}")
    mlflow.set_registry_uri(f"databricks-uc://{DATABRICKS_PROFILE}")


@pytest.fixture(scope="module")
def spark():
    return DatabricksSession.builder.profile(DATABRICKS_PROFILE).serverless(True).getOrCreate()


@pytest.fixture(scope="module")
def trained_model():
    """Whatever model is currently tagged 'latest-model' -- i.e. the real,
    currently-deployed model, not a freshly trained throwaway one."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(name=MODEL_NAME, alias="latest-model")
    return mlflow.sklearn.load_model(f"models:/{version.model_id}")


@pytest.fixture(scope="module")
def train_data(spark):
    df = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.train_set").toPandas()
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    return X, y


@pytest.fixture(scope="module")
def test_data(spark):
    df = spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.test_set").toPandas()
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]
    return X, y


@pytest.fixture(scope="module")
def basic_model_config():
    """A config for actually running BasicModel.train() in the timing test --
    separate from trained_model, which loads the already-registered model."""
    return SimpleNamespace(
        num_features=NUM_FEATURES,
        cat_features=CAT_FEATURES,
        target=TARGET,
        parameters={"n_estimators": 50, "max_depth": 5, "verbose": -1},
        catalog_name=CATALOG_NAME,
        schema_name=SCHEMA_NAME,
        experiment_name_basic="/Shared/marvel-characters-training-tests",
    )


@pytest.fixture(scope="module")
def basic_model_tags():
    from marvel_characters.config import Tags

    return Tags(git_sha="timing-test", branch="timing-test")
