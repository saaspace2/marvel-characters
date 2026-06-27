"""Shared fixtures for data quality tests.

Data Quality Tests are READ-ONLY -- they check the real, current data sitting
in your dev tables, the same data your actual pipeline produced. Unlike
integration tests, there's no throwaway sandbox schema needed here, since
nothing gets written or modified.
"""

import pytest
from databricks.connect import DatabricksSession

CATALOG_NAME = "mlops_dev"
SCHEMA_NAME = "marvel_characters"

# Matches the profile name shown by `databricks auth profiles`.
DATABRICKS_PROFILE = "dbc-34ada41e-0aa1"


@pytest.fixture(scope="module")
def spark():
    return DatabricksSession.builder.profile(DATABRICKS_PROFILE).serverless(True).getOrCreate()


@pytest.fixture(scope="module")
def train_data(spark):
    return spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.train_set").toPandas()


@pytest.fixture(scope="module")
def test_data(spark):
    return spark.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.test_set").toPandas()
