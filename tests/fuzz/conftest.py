"""Shared fixtures for Fuzz Tests.

Fuzz tests throw random, unexpected, and invalid inputs at the Marvel
Characters serving endpoint to discover crashes and unexpected behaviour.

GOLDEN RULE
-----------
The system must NEVER return HTTP 500 regardless of what input is sent.
    200 = worked correctly
    400 = bad input was rejected cleanly  (acceptable)
    500 = server crashed                  (always a bug)

IMPORTANT -- why this conftest serializes JSON manually
-------------------------------------------------------
The whole point of fuzzing is to send values like inf, -inf, nan, and raw
bytes to the SERVER and check the server does not crash.  But Python's
`requests` library (and the stdlib json module) refuse to serialize those
values and raise a ValueError on the CLIENT side before the request is ever
sent.  That client-side crash is not a real test result -- it just means the
test harness gave up.

To fuzz properly we must bypass that client-side guard and put the bad bytes
on the wire ourselves, then let the SERVER decide how to handle them.  We do
that by:
  1. Serializing with json.dumps(..., allow_nan=True) which emits the
     non-standard tokens Infinity, -Infinity, NaN (most servers accept these).
  2. Falling back to a safe sanitized payload if a value is truly
     unserializable (e.g. raw bytes), converting it to its string repr so
     the request still reaches the server.
  3. Sending the result with data=<bytes> instead of json=<obj> so requests
     does not re-validate it.

PREREQUISITES
-------------
Install Hypothesis before running:
    pip install hypothesis

Set environment variables:
    export ENDPOINT_URL="https://<workspace>.azuredatabricks.net/serving-endpoints/<name>/invocations"
    export DBR_TOKEN="<your-databricks-personal-access-token>"

Run:
    pytest tests/fuzz/ -v
    pytest tests/fuzz/ -v --hypothesis-seed=0   # reproducible run
"""

import json
import os

import pytest
import requests


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.exit(
            f"\n\nMissing required environment variable: {name}\n"
            f"Set it before running fuzz tests:\n"
            f"    export {name}='<value>'\n",
            returncode=1,
        )
    return value


# ---------------------------------------------------------------------------
# A valid baseline record -- used as the starting point for every fuzz mutation.
# Fuzz tests replace ONE or MORE fields with bad values while keeping the
# rest valid so we can isolate exactly which field caused a crash.
# ---------------------------------------------------------------------------

VALID_RECORD = {
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

ALL_FIELDS = list(VALID_RECORD.keys())
NUMERIC_FIELDS = ["Height", "Weight"]
INTEGER_FIELDS = ["Teams", "Magic", "Mutant"]
STRING_FIELDS  = ["Universe", "Identity", "Gender", "Marital_Status", "Origin"]


# ---------------------------------------------------------------------------
# Safe serialization helpers
# ---------------------------------------------------------------------------

def _make_json_safe(obj):
    """Recursively convert values the stdlib json encoder cannot handle into
    something serializable, so the fuzz payload can still be sent to the
    server instead of crashing the test client.

    - bytes / bytearray  -> their string repr (so they still travel as text)
    - sets / tuples       -> lists
    - anything exotic      -> its string repr

    NOTE: inf / -inf / nan are intentionally left alone here -- they are
    handled by json.dumps(allow_nan=True) in _serialize() below, which emits
    Infinity / -Infinity / NaN tokens that real serving endpoints accept.
    """
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return repr(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _serialize(payload: dict) -> bytes:
    """Serialize a payload to JSON bytes, tolerating inf/-inf/nan/bytes.

    Stage 1: json.dumps(allow_nan=True) emits the non-standard tokens
             Infinity / -Infinity / NaN which Databricks Model Serving and
             most JSON parsers accept -- letting us fuzz the server with
             those values instead of crashing the client.
    Stage 2: if that still fails (e.g. raw bytes present), sanitize the
             payload with _make_json_safe and serialize again.
    """
    try:
        return json.dumps(payload, allow_nan=True).encode("utf-8")
    except (TypeError, ValueError):
        safe = _make_json_safe(payload)
        return json.dumps(safe, allow_nan=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def endpoint_url() -> str:
    return _require_env("ENDPOINT_URL")


@pytest.fixture(scope="session")
def dbr_token() -> str:
    return _require_env("DBR_TOKEN")


@pytest.fixture(scope="session")
def valid_record() -> dict:
    """A single known-good record. Fuzz tests mutate copies of this."""
    return VALID_RECORD.copy()


@pytest.fixture(scope="session")
def call_endpoint(endpoint_url, dbr_token):
    """Session-scoped callable: posts a list of records, returns (status, text).

    Accepts both a single dict (wraps it) and a list of dicts so tests
    can pass either format without thinking about it.

    Unlike a naive requests.post(json=...) call, this serializes the body
    MANUALLY with allow_nan=True and sends it via data=<bytes>. That means
    inf / -inf / nan / raw bytes actually reach the server (which is the
    whole point of fuzzing) instead of crashing the test client.
    """
    headers = {
        "Authorization": f"Bearer {dbr_token}",
        "Content-Type": "application/json",
    }

    def _call(record, timeout: int = 30) -> tuple[int, str]:
        payload = record if isinstance(record, list) else [record]
        body = _serialize({"dataframe_records": payload})
        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                data=body,          # raw bytes -> requests will NOT re-validate
                timeout=timeout,
            )
            return response.status_code, response.text
        except requests.exceptions.Timeout:
            return 408, "Request timed out"
        except requests.exceptions.ConnectionError as exc:
            return 503, f"Connection error: {exc}"

    return _call
