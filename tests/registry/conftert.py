"""Shared fixtures for Model Registry tests.

These tests are READ-ONLY against the real Unity Catalog model registry --
no writes, no risk to existing registered models. They verify that models
your training pipeline registered actually meet the bar: they exist, have
the right metadata, and can actually be loaded and used.
"""

import mlflow
import pytest
from mlflow import MlflowClient

# Matches the profile name shown by `databricks auth profiles`.
DATABRICKS_PROFILE = "dbc-34ada41e-0aa1"


@pytest.fixture(scope="module", autouse=True)
def _configure_mlflow():
    """Point MLflow at the real Databricks Unity Catalog registry.

    The profile is embedded directly in the URI string (databricks://profile,
    databricks-uc://profile) rather than relying on the DATABRICKS_CONFIG_PROFILE
    env var alone -- env var propagation timing relative to fixture/import
    order isn't guaranteed, but an explicit URI always resolves correctly.
    """
    mlflow.set_tracking_uri(f"databricks://{DATABRICKS_PROFILE}")
    mlflow.set_registry_uri(f"databricks-uc://{DATABRICKS_PROFILE}")


@pytest.fixture(scope="module")
def client():
    return MlflowClient()
