"""Shared fixtures for integration tests.

These tests connect to a REAL Databricks serverless cluster via Databricks
Connect, and read/write REAL Delta tables. To avoid ANY risk of touching your
actual dev train_set/test_set tables, every test session creates its own
uniquely-named schema inside the mlops_dev catalog. That schema (and
everything in it) is dropped automatically at the end of the session, even
if a test fails partway through.

Requires:
- Local auth already configured for the workspace in databricks.yml
  (e.g. `databricks auth login`, or DATABRICKS_HOST/DATABRICKS_TOKEN env vars)
- CREATE SCHEMA privilege on the mlops_dev catalog
"""

import uuid

import pandas as pd
import pytest
from databricks.connect import DatabricksSession

from marvel_characters.config import ProjectConfig, Tags

CATALOG_NAME = "mlops_dev"

# Matches the profile name shown by `databricks auth profiles`.
# If you ever rename it (e.g. to "DEFAULT"), this can go back to no .profile() call.
DATABRICKS_PROFILE = "dbc-34ada41e-0aa1"


@pytest.fixture(scope="session")
def spark():
    """Real Spark session on the connected serverless cluster."""
    return DatabricksSession.builder.profile(DATABRICKS_PROFILE).serverless(True).getOrCreate()


@pytest.fixture(scope="session")
def test_schema_name(spark):
    """Create a throwaway schema for this test session, then drop it afterward.

    This guarantees integration tests NEVER touch the real
    mlops_dev.marvel_characters.train_set / test_set tables used by your
    actual pipeline, regardless of what save_to_catalog() writes.
    """
    schema_name = f"marvel_characters_inttest_{uuid.uuid4().hex[:8]}"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{schema_name}")

    yield schema_name

    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG_NAME}.{schema_name} CASCADE")


@pytest.fixture
def integration_config(test_schema_name):
    """ProjectConfig pointing at the isolated test schema, never real dev data."""
    return ProjectConfig(
        num_features=["Height", "Weight"],
        cat_features=[
            "Universe",
            "Identity",
            "Gender",
            "Marital_Status",
            "Teams",
            "Origin",
            "Magic",
            "Mutant",
        ],
        target="Alive",
        catalog_name=CATALOG_NAME,
        schema_name=test_schema_name,
        parameters={"n_estimators": 10, "max_depth": 3, "verbose": -1},
        experiment_name_basic="/Shared/marvel-characters-integration-tests",
        experiment_name_custom=None,
    )


@pytest.fixture
def integration_tags():
    return Tags(git_sha="integration-test-sha", branch="integration-test")


@pytest.fixture
def raw_marvel_df():
    """A small but realistic raw dataset, same shape as the real source data."""
    return pd.DataFrame(
        {
            "Height (m)": [1.8, 1.6, 1.9, 1.7, 1.75, 1.65, 1.85, 1.78, 1.6, 1.95],
            "Weight (kg)": [90.0, 55.0, 100.0, 70.0, 80.0, 60.0, 95.0, 85.0, 58.0, 110.0],
            "Universe": ["Earth-616"] * 10,
            "Teams": [
                "Avengers", None, "X-Men", "Avengers", None,
                "X-Men", "Avengers", None, "X-Men", "Avengers",
            ],
            "Origin": [
                "Human mutate", "Asgardian god", "Human", "Alien race", "Human mutant",
                "Symbiote host", "Robot built", "Human", "Cosmic Being entity", "Human",
            ],
            "Identity": [
                "Public", "Secret", "Public", "Secret", "Public",
                "Secret", "Public", "Secret", "Public", "Secret",
            ],
            "Gender": [
                "Male", "Female", "Male", "Female", "Male",
                "Female", "Male", "Female", "Male", "Female",
            ],
            "Marital Status": [
                "Single", "Widow", "Married", "Single", "Engaged",
                "Single", "Married", "Single", "Widow", "Married",
            ],
            "Alive": [
                "Alive", "Dead", "Alive", "Alive", "Dead",
                "Alive", "Alive", "Dead", "Alive", "Alive",
            ],
            "PageID": list(range(101, 111)),
        }
    )
