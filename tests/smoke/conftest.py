"""Shared fixtures for Smoke Tests.

Smoke tests are the FIRST tests that run immediately after every deployment.
They confirm the serving endpoint is alive and responding before any other
test suite runs.  If smoke tests fail, there is no point running integration,
regression, or performance tests -- the system is simply not up.

These tests hit the REAL Databricks Model Serving endpoint via HTTP.
Set the following environment variables before running:

    export ENDPOINT_URL="https://<workspace>.azuredatabricks.net/serving-endpoints/<name>/invocations"
    export DBR_TOKEN="<your-databricks-personal-access-token>"

Run:
    pytest tests/serving/smoke/ -v

Expected outcome on a healthy deployment:
    All tests PASS in under 5 seconds combined.
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
            f"Set it before running smoke tests:\n"
            f"    export {name}='<value>'\n",
            returncode=1,
        )
    return value


# ---------------------------------------------------------------------------
# Sample payload — one valid Marvel character record
# ---------------------------------------------------------------------------

SAMPLE_RECORD = [
    {
        "Height":        1.75,
        "Weight":        70.0,
        "Universe":      "Earth-616",
        "Identity":      "Public",
        "Gender":        "Male",
        "Marital_Status": "Single",
        "Teams":         1,
        "Origin":        "Human",
        "Magic":         0,
        "Mutant":        0,
    }
]


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
    """A single valid input row that the model should accept and score."""
    return SAMPLE_RECORD


@pytest.fixture(scope="session")
def call_endpoint(endpoint_url, dbr_token):
    """Return a callable that posts a payload to the endpoint and returns
    (status_code, response_text).

    Using a session-scoped fixture means the URL and token are resolved once
    per test session instead of on every individual test call.
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
def live_response(call_endpoint, sample_record):
    """Make ONE real HTTP call and cache the result for the whole session.

    Multiple smoke tests inspect the same response (status code, body,
    JSON validity, required keys).  Caching avoids hammering the endpoint
    with identical requests and makes the smoke suite as fast as possible.
    """
    status, text = call_endpoint(sample_record)
    return status, text
