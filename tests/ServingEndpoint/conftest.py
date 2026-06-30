"""Shared fixtures for Serving Endpoint Tests.

Serving endpoint tests go DEEPER than smoke tests.
Smoke tests ask: "Is the system alive?"
Serving tests ask: "Is the system correct, fast, and robust?"

These tests hit the REAL Databricks Model Serving endpoint via HTTP.
Set the following environment variables before running:

    export ENDPOINT_URL="https://<workspace>.cloud.databricks.com/serving-endpoints/marvel-character-model-serving/invocations"
    export DBR_TOKEN="<your-databricks-personal-access-token>"

Run:
    pytest tests/serving/ -v

Expected outcome on a healthy deployment:
    All tests PASS in under 60 seconds combined.

Precondition: Smoke tests must already be passing.
If smoke tests fail, do not run serving tests — the endpoint is not alive.
"""

import json
import os
import time

import pytest
import requests


# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Read a required environment variable, fail clearly if it is missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.exit(
            f"\n\nMissing required environment variable: {name}\n"
            f"Set it before running serving tests:\n"
            f"    export {name}='<value>'\n"
            f"\nOn Windows PowerShell:\n"
            f"    $env:{name} = '<value>'\n",
            returncode=1,
        )
    return value


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

# A single valid Marvel character record — the happy path
SAMPLE_RECORD = [
    {
        "Height":         1.75,
        "Weight":         70.0,
        "Universe":       "Earth-616",
        "Identity":       "Public",
        "Gender":         "Male",
        "Marital_Status": "Single",
        "Teams":          1,
        "Origin":         "Human",
        "Magic":          0,
        "Mutant":         0,
    }
]

# A batch of 3 records — tests that the endpoint handles multiple rows
BATCH_RECORDS = [
    {
        "Height": 1.75, "Weight": 70.0, "Universe": "Earth-616",
        "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
        "Teams": 1, "Origin": "Human", "Magic": 0, "Mutant": 0,
    },
    {
        "Height": 1.90, "Weight": 95.0, "Universe": "Earth-616",
        "Identity": "Secret", "Gender": "Male", "Marital_Status": "Married",
        "Teams": 0, "Origin": "Mutant", "Magic": 0, "Mutant": 1,
    },
    {
        "Height": 1.65, "Weight": 55.0, "Universe": "Earth-199999",
        "Identity": "Public", "Gender": "Female", "Marital_Status": "Single",
        "Teams": 1, "Origin": "Human", "Magic": 1, "Mutant": 0,
    },
]

# A record with all null values — tests graceful null handling
NULL_RECORD = [
    {
        "Height": None, "Weight": None, "Universe": None,
        "Identity": None, "Gender": None, "Marital_Status": None,
        "Teams": None, "Origin": None, "Magic": None, "Mutant": None,
    }
]

# Edge case records for boundary testing
EDGE_CASE_RECORDS = {
    "zero_height_weight": [{
        "Height": 0.0, "Weight": 0.0, "Universe": "Earth-616",
        "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
        "Teams": 0, "Origin": "Human", "Magic": 0, "Mutant": 0,
    }],
    "very_large_values": [{
        "Height": 999.9, "Weight": 999.9, "Universe": "Earth-616",
        "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
        "Teams": 1, "Origin": "Cosmic Being", "Magic": 1, "Mutant": 0,
    }],
    "unknown_universe": [{
        "Height": 1.75, "Weight": 70.0, "Universe": "Earth-99999",
        "Identity": "Unknown", "Gender": "Other", "Marital_Status": "Unknown",
        "Teams": 0, "Origin": "Other", "Magic": 0, "Mutant": 0,
    }],
    "all_binary_flags_on": [{
        "Height": 1.75, "Weight": 70.0, "Universe": "Earth-616",
        "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
        "Teams": 1, "Origin": "Human", "Magic": 1, "Mutant": 1,
    }],
    "all_binary_flags_off": [{
        "Height": 1.75, "Weight": 70.0, "Universe": "Earth-616",
        "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
        "Teams": 0, "Origin": "Human", "Magic": 0, "Mutant": 0,
    }],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def endpoint_url() -> str:
    """The full invocation URL for the Databricks Model Serving endpoint."""
    return _require_env("ENDPOINT_URL")


@pytest.fixture(scope="session")
def dbr_token() -> str:
    """Databricks personal access token used for Bearer auth."""
    return _require_env("DBR_TOKEN")


@pytest.fixture(scope="session")
def sample_record() -> list[dict]:
    """A single valid input row."""
    return SAMPLE_RECORD


@pytest.fixture(scope="session")
def batch_records() -> list[dict]:
    """Three valid input rows for batch testing."""
    return BATCH_RECORDS


@pytest.fixture(scope="session")
def null_record() -> list[dict]:
    """A record with all null values."""
    return NULL_RECORD


@pytest.fixture(scope="session")
def edge_case_records() -> dict:
    """Dictionary of edge case records keyed by scenario name."""
    return EDGE_CASE_RECORDS


@pytest.fixture(scope="session")
def call_endpoint(endpoint_url, dbr_token):
    """Return a callable that posts a payload and returns (status_code, response_text).

    session-scoped so the URL and token are resolved once per test run.
    """
    def _call(record: list[dict], timeout: int = 30) -> tuple[int, str]:
        response = requests.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {dbr_token}"},
            json={"dataframe_records": record},
            timeout=timeout,
        )
        return response.status_code, response.text

    return _call


@pytest.fixture(scope="session")
def live_response(call_endpoint, sample_record) -> tuple[int, str]:
    """One cached HTTP call for tests that inspect the same response.

    Avoids hitting the endpoint multiple times for tests that only
    need to inspect the shape/content of a single valid response.
    """
    return call_endpoint(sample_record)


@pytest.fixture(scope="session")
def parsed_response(live_response) -> dict:
    """The parsed JSON body of the cached live response."""
    _, response_text = live_response
    return json.loads(response_text)


@pytest.fixture(scope="session")
def predictions(parsed_response) -> dict:
    """The 'predictions' value from the cached response."""
    return parsed_response.get("predictions", {})
