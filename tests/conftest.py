"""SINGLE merged conftest for ALL test folders.

Place at:  tests/conftest.py   (ONE file, nothing else needed)

This file contains BOTH:
  - the shared FIXTURES (call_endpoint, valid_record, record_bank, ...)
  - the helper FUNCTIONS (extract_prediction, normalise_prediction, ...)
  - the shared DATA (VALID_RECORD, RECORD_BANK, field lists)

pytest automatically shares everything defined here with every subfolder
(contract/, fuzz/, performance/, regression/, ...). You do NOT need a
conftest.py in each folder, and you do NOT need a separate helpers.py.

HOW TEST FILES IMPORT THE HELPERS
---------------------------------
In any test file, import the helper functions like this:

    from tests.conftest import extract_prediction, normalise_prediction

OR, if you run pytest from the project root (recommended), this also works
because the root tests/ conftest is importable as a top-level module:

    from conftest import extract_prediction, normalise_prediction

Both styles are supported by the sys.path line below.

ENV VARS
--------
Required:  ENDPOINT_URL, DBR_TOKEN
Optional:  ENDPOINT_URL_SHADOW, ENDPOINT_URL_CANARY, ENDPOINT_URL_A,
           ENDPOINT_URL_B, REFERENCE_STATS_FILE
"""

import json
import os
import sys
import time

import pytest
import requests

# Make BOTH the tests/ dir and its parent importable, so every folder can do
# `from conftest import ...` and `from tests.conftest import ...`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                      # tests/
sys.path.insert(0, os.path.dirname(_THIS_DIR))     # project root


# ===========================================================================
# SHARED DATA  (was in helpers.py)
# ===========================================================================

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

ALL_FIELDS     = list(VALID_RECORD.keys())
NUMERIC_FIELDS = ["Height", "Weight"]
INTEGER_FIELDS = ["Teams", "Magic", "Mutant"]
STRING_FIELDS  = ["Universe", "Identity", "Gender", "Marital_Status", "Origin"]

RECORD_BANK = {
    "case_001": {**VALID_RECORD, "Height": 1.80, "Weight": 90.0,  "Origin": "Mutant",       "Mutant": 1},
    "case_002": {**VALID_RECORD, "Height": 1.60, "Weight": 55.0,  "Origin": "Human",        "Gender": "Female"},
    "case_003": {**VALID_RECORD, "Height": 2.00, "Weight": 120.0, "Origin": "Asgardian",    "Magic": 1},
    "case_004": {**VALID_RECORD, "Height": 1.70, "Weight": 68.0,  "Origin": "Alien",        "Universe": "Earth-1610"},
    "case_005": {**VALID_RECORD, "Height": 1.65, "Weight": 60.0,  "Origin": "Symbiote",     "Identity": "Secret"},
    "case_006": {**VALID_RECORD, "Height": 1.90, "Weight": 100.0, "Origin": "Robot",        "Marital_Status": "Married"},
    "case_007": {**VALID_RECORD, "Height": 1.55, "Weight": 50.0,  "Origin": "Human",        "Teams": 0},
    "case_008": {**VALID_RECORD, "Height": 1.85, "Weight": 95.0,  "Origin": "Cosmic Being", "Magic": 1},
}


# ===========================================================================
# HELPER FUNCTIONS  (was in helpers.py)
# Importable via:  from conftest import extract_prediction, normalise_prediction
# ===========================================================================

def extract_prediction(body):
    """Pull a single prediction out of any response shape.
    Returns 0/1, a string label ('alive'/'dead'), or None."""
    preds = body.get("predictions", body) if isinstance(body, dict) else body
    if isinstance(preds, dict):
        for key in ("Prediction", "prediction", "Survival prediction"):
            if key in preds:
                val = preds[key]
                return val[0] if isinstance(val, list) and val else val
        vals = list(preds.values())
        if len(vals) == 1:
            v = vals[0]
            return v[0] if isinstance(v, list) and v else v
    if isinstance(preds, list):
        return preds[0] if preds else None
    return preds


def normalise_prediction(value):
    """Map any prediction representation to canonical 0/1.
    'alive' -> 1, 'dead' -> 0, ints stay as-is."""
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "alive":
            return 1
        if low == "dead":
            return 0
    return value


# ===========================================================================
# CLI flags + markers (declared ONCE at the root)
# ===========================================================================

def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="Regression: capture current predictions as golden baseline.")
    parser.addoption("--update-snapshot", action="store_true", default=False,
                     help="Snapshot: capture current response shape as the snapshot.")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running concurrent/load tests")
    config.addinivalue_line("markers", "integration: requires a live Databricks endpoint")


@pytest.fixture(scope="session")
def update_golden(request):
    return request.config.getoption("--update-golden", default=False)


@pytest.fixture(scope="session")
def update_snapshot(request):
    return request.config.getoption("--update-snapshot", default=False)


# ===========================================================================
# Env helpers
# ===========================================================================

def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.exit(f"\n\nMissing required environment variable: {name}\n"
                    f"    export {name}='<value>'\n", returncode=1)
    return value


def _optional_env(name: str):
    return os.environ.get(name, "").strip() or None


# ===========================================================================
# Core fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def endpoint_url() -> str:
    return _require_env("ENDPOINT_URL")


@pytest.fixture(scope="session")
def dbr_token() -> str:
    return _require_env("DBR_TOKEN")


@pytest.fixture(scope="session")
def valid_record() -> dict:
    return VALID_RECORD.copy()


@pytest.fixture(scope="session")
def record_bank() -> dict:
    return {k: v.copy() for k, v in RECORD_BANK.items()}


def _post(url, token, records, timeout=30):
    """POST -> (status_code, parsed_body_or_text, elapsed_ms).

    Serializes with allow_nan=True and sends raw bytes so fuzz tests can send
    inf/-inf/nan without the client refusing to serialize them.
    """
    start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=json.dumps({"dataframe_records": records}, allow_nan=True).encode("utf-8"),
            timeout=timeout,
        )
        elapsed = (time.perf_counter() - start) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {"_raw": resp.text}
        return resp.status_code, body, elapsed
    except requests.exceptions.Timeout:
        return 408, {"error": "timeout"}, (time.perf_counter() - start) * 1000
    except requests.exceptions.ConnectionError as exc:
        return 503, {"error": str(exc)}, (time.perf_counter() - start) * 1000


@pytest.fixture(scope="session")
def call_endpoint(endpoint_url, dbr_token):
    """Primary endpoint caller -> (status, body, elapsed_ms).
    New tests should unpack as:  status, body, _ = call_endpoint(rec)
    """
    def _call(record, timeout: int = 30):
        records = record if isinstance(record, list) else [record]
        return _post(endpoint_url, dbr_token, records, timeout)
    return _call


@pytest.fixture(scope="session")
def call_named_endpoint(dbr_token):
    """Caller for an arbitrary endpoint URL -> (status, body, elapsed_ms)."""
    def _call(url, record, timeout: int = 30):
        records = record if isinstance(record, list) else [record]
        return _post(url, dbr_token, records, timeout)
    return _call


# ===========================================================================
# Optional second-endpoint fixtures (suite skips if unset)
# ===========================================================================

@pytest.fixture(scope="session")
def shadow_url():
    url = _optional_env("ENDPOINT_URL_SHADOW")
    if not url:
        pytest.skip("ENDPOINT_URL_SHADOW not set - shadow tests skipped")
    return url


@pytest.fixture(scope="session")
def canary_url():
    url = _optional_env("ENDPOINT_URL_CANARY")
    if not url:
        pytest.skip("ENDPOINT_URL_CANARY not set - canary tests skipped")
    return url


@pytest.fixture(scope="session")
def model_a_url():
    url = _optional_env("ENDPOINT_URL_A")
    if not url:
        pytest.skip("ENDPOINT_URL_A not set - A/B tests skipped")
    return url


@pytest.fixture(scope="session")
def model_b_url():
    url = _optional_env("ENDPOINT_URL_B")
    if not url:
        pytest.skip("ENDPOINT_URL_B not set - A/B tests skipped")
    return url
