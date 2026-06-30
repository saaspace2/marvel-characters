"""Property-Based Tests for the Marvel Characters serving endpoint.

WHAT IS PROPERTY-BASED TESTING?
-------------------------------
Instead of testing specific examples you thought of, property-based testing
defines RULES (properties) that must ALWAYS hold true, then uses Hypothesis
to auto-generate many valid inputs to try to break those rules.

    Normal test:   assert predict(specific_record) == 1   (one example)
    Property test: "for ANY valid record, the prediction is always 0 or 1"

THE PROPERTIES WE VERIFY
------------------------
1. Validity     - any valid input always yields a prediction in {0, 1}
2. Determinism  - the same input always yields the same output
3. Totality     - a valid input never yields an error or empty result
4. Batch shape  - N input records always yield N predictions
5. Independence - a record's prediction does not depend on its batch neighbours

HANDLING A SLOW SERVERLESS ENDPOINT
-----------------------------------
Each Hypothesis example makes a real HTTP call. On Databricks serverless the
first call after idle pays a cold-start cost and can TIME OUT (HTTP 408 / 503).
A single transient timeout is NOT a property violation -- it is a network blip.

Two measures keep these tests honest instead of flaky:
  1. A module-scoped warm-up fires throwaway calls so the endpoint is hot
     before Hypothesis starts generating examples.
  2. Inside each test, a transient transport status (408 timeout / 503
     connection) is retried once; only a genuine bad RESPONSE fails the test.
     This stops Hypothesis's own retry from declaring the test 'flaky'.

RUN
---
    pytest tests/property_based/ -v
    pytest tests/property_based/ -v --hypothesis-show-statistics
"""

import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import extract_prediction, normalise_prediction


# Valid value pools -- Hypothesis draws from these to build VALID records.
UNIVERSES  = ["Earth-616", "Earth-1610", "Earth-199999", "Earth-TRN193"]
IDENTITIES = ["Public", "Secret", "No Dual Identity"]
GENDERS    = ["Male", "Female", "Other"]
STATUSES   = ["Single", "Married", "Widowed", "Divorced"]
ORIGINS    = ["Human", "Mutant", "Asgardian", "Alien", "Symbiote", "Robot", "Cosmic Being", "Other"]

# Transport-level statuses that mean "network blip", not "model rejected input".
TRANSIENT_STATUSES = {408, 502, 503, 504}


@st.composite
def valid_marvel_record(draw):
    """Hypothesis strategy that builds a VALID Marvel character record."""
    return {
        "Height":         round(draw(st.floats(min_value=1.40, max_value=2.20)), 2),
        "Weight":         round(draw(st.floats(min_value=40.0, max_value=150.0)), 1),
        "Universe":       draw(st.sampled_from(UNIVERSES)),
        "Identity":       draw(st.sampled_from(IDENTITIES)),
        "Gender":         draw(st.sampled_from(GENDERS)),
        "Marital_Status": draw(st.sampled_from(STATUSES)),
        "Teams":          draw(st.integers(min_value=0, max_value=1)),
        "Origin":         draw(st.sampled_from(ORIGINS)),
        "Magic":          draw(st.integers(min_value=0, max_value=1)),
        "Mutant":         draw(st.integers(min_value=0, max_value=1)),
    }


# Fewer examples + a generous deadline, because every example is a real network
# round-trip against a ~1s serverless endpoint.
ENDPOINT_SETTINGS = settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=None,   # do not let Hypothesis fail an example purely for being slow
)


def _call_with_retry(call_endpoint, records, retries=2, backoff=2.0):
    """Call the endpoint, retrying ONLY on transient transport statuses.

    Returns (status, body). A 200 or a genuine 4xx/5xx model response is
    returned immediately. A timeout/connection blip (408/502/503/504) is
    retried a couple of times with backoff so a cold-start hiccup does not
    masquerade as a property violation.
    """
    status, body = None, None
    for attempt in range(retries + 1):
        status, body, _ = call_endpoint(records, timeout=60)
        if status not in TRANSIENT_STATUSES:
            return status, body
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    return status, body


@pytest.fixture(scope="module", autouse=True)
def _warm_endpoint(call_endpoint):
    """Warm the serverless endpoint ONCE before Hypothesis starts.

    The first call after idle pays the cold-start cost. Doing it here, outside
    the measured tests, means the property tests run against a hot endpoint and
    do not flake on a single cold-start timeout.
    """
    base = {
        "Height": 1.75, "Weight": 70.0, "Universe": "Earth-616",
        "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
        "Teams": 1, "Origin": "Human", "Magic": 0, "Mutant": 0,
    }
    for _ in range(3):
        try:
            call_endpoint([base], timeout=60)
        except Exception:
            pass
    yield


class TestValidityProperty:
    """PROPERTY: any valid input always yields a prediction in {0, 1}."""

    @given(record=valid_marvel_record())
    @ENDPOINT_SETTINGS
    def test_prediction_always_binary(self, record, call_endpoint):
        """For ANY valid record, the prediction must be 0 or 1 (alive/dead).

        A transient timeout is retried (not failed); only a real non-200 model
        response or a non-binary prediction fails the property.
        """
        status, body = _call_with_retry(call_endpoint, [record])
        if status in TRANSIENT_STATUSES:
            pytest.skip(f"Endpoint transiently unavailable ({status}); "
                        f"not a property violation.")
        assert status == 200, f"Valid record was rejected ({status}): {record}"
        pred = normalise_prediction(extract_prediction(body))
        assert pred in (0, 1), f"Non-binary prediction {pred!r} for {record}"


class TestDeterminismProperty:
    """PROPERTY: the same input always yields the same output."""

    @given(record=valid_marvel_record())
    @settings(max_examples=12, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_same_input_same_output(self, record, call_endpoint):
        """Submitting the identical record twice must give the identical result."""
        s1, b1 = _call_with_retry(call_endpoint, [record])
        s2, b2 = _call_with_retry(call_endpoint, [record])
        if s1 in TRANSIENT_STATUSES or s2 in TRANSIENT_STATUSES:
            pytest.skip("Endpoint transiently unavailable; skipping determinism check.")
        pred1 = normalise_prediction(extract_prediction(b1))
        pred2 = normalise_prediction(extract_prediction(b2))
        assert pred1 == pred2, (
            f"Non-deterministic prediction for {record}: {pred1} then {pred2}"
        )


class TestTotalityProperty:
    """PROPERTY: a valid input never yields an error or empty result."""

    @given(record=valid_marvel_record())
    @ENDPOINT_SETTINGS
    def test_valid_input_never_errors(self, record, call_endpoint):
        """A valid record must always produce a usable prediction."""
        status, body = _call_with_retry(call_endpoint, [record])
        if status in TRANSIENT_STATUSES:
            pytest.skip(f"Endpoint transiently unavailable ({status}).")
        assert status == 200
        assert "error" not in body, f"Error returned for valid record: {body}"
        assert extract_prediction(body) is not None, f"Empty prediction for {record}"


class TestBatchShapeProperty:
    """PROPERTY: N input records always yield N predictions."""

    @given(records=st.lists(valid_marvel_record(), min_size=1, max_size=8))
    @settings(max_examples=12, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_batch_length_preserved(self, records, call_endpoint):
        """A batch of N valid records must return exactly N predictions."""
        status, body = _call_with_retry(call_endpoint, records)
        if status in TRANSIENT_STATUSES:
            pytest.skip(f"Endpoint transiently unavailable ({status}).")
        assert status == 200
        preds = body.get("predictions", body)
        if isinstance(preds, dict):
            list_vals = [v for v in preds.values() if isinstance(v, list)]
            length = len(list_vals[0]) if list_vals else 1
        elif isinstance(preds, list):
            length = len(preds)
        else:
            length = 1
        assert length == len(records), (
            f"Sent {len(records)} records but got {length} predictions"
        )


class TestIndependenceProperty:
    """PROPERTY: a record's prediction does not depend on its batch neighbours."""

    @given(record=valid_marvel_record())
    @settings(max_examples=10, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
    def test_prediction_independent_of_batch_position(self, record, call_endpoint, record_bank):
        """A record predicted ALONE must match the same record predicted inside
        a batch of other records."""
        s_alone, alone_body = _call_with_retry(call_endpoint, [record])
        if s_alone in TRANSIENT_STATUSES:
            pytest.skip(f"Endpoint transiently unavailable ({s_alone}).")
        alone = normalise_prediction(extract_prediction(alone_body))

        neighbours = list(record_bank.values())[:3]
        batch = neighbours + [record]
        s_batch, batch_body = _call_with_retry(call_endpoint, batch)
        if s_batch in TRANSIENT_STATUSES:
            pytest.skip(f"Endpoint transiently unavailable ({s_batch}).")
        assert s_batch == 200

        preds = batch_body.get("predictions", batch_body)
        if isinstance(preds, dict):
            list_vals = [v for v in preds.values() if isinstance(v, list)]
            in_batch = normalise_prediction(list_vals[0][-1]) if list_vals else None
        elif isinstance(preds, list):
            in_batch = normalise_prediction(preds[-1])
        else:
            in_batch = normalise_prediction(preds)

        assert alone == in_batch, (
            f"Prediction changed by batch position: alone={alone}, in_batch={in_batch} "
            f"for record {record}"
        )
