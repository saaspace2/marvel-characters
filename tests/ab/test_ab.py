"""A/B Tests for the Marvel Characters serving endpoint.

WHAT IS A/B TESTING?
--------------------
A/B testing validates that traffic is correctly split between two model
versions and that both versions return valid responses.

    Analogy: a supermarket puts packaging design A on the left shelf and B on
    the right. Half the customers see A, half see B. After a month they compare
    sales. In ML, A/B testing compares two model versions on real traffic to
    learn which performs better.

WHAT WE VERIFY
--------------
1. Both models receive traffic (neither gets 0%).
2. The split is approximately as configured (e.g. ~50/50).
3. Routing is DETERMINISTIC -- the same input always goes to the same model
   (otherwise one user could get two different answers).
4. Both models return valid predictions.
5. The response identifies which model served it (needed for analysis).

TWO MODES
---------
A) Single A/B endpoint that internally splits traffic and returns a 'model'
   field saying which arm served the request. (Tested via call_endpoint.)
B) Two separate endpoints ENDPOINT_URL_A and ENDPOINT_URL_B that you compare
   directly. (Tested via model_a_url / model_b_url fixtures.)

RUN
---
    pytest tests/advanced/test_ab.py -v
"""

import hashlib

import pytest

from conftest import extract_prediction, normalise_prediction


def _model_of(body: dict):
    """Extract which model arm served a response, if the endpoint reports it."""
    if isinstance(body, dict):
        for key in ("model", "model_name", "variant", "arm"):
            if key in body:
                return body[key]
    return None


# ===========================================================================
# MODE A — single endpoint that internally A/B-splits and tags each response
# ===========================================================================

class TestSingleEndpointSplit:
    """Tests for an endpoint that splits internally and returns a 'model' field."""

    def test_response_identifies_serving_model(self, call_endpoint, valid_record):
        """Each response must say which model served it.

        Without a model identifier, you cannot attribute outcomes to arms,
        so the entire A/B experiment is unanalysable.
        """
        _, body, _ = call_endpoint([valid_record])
        model = _model_of(body)
        if model is None:
            pytest.skip("Endpoint does not report a 'model' field; use MODE B "
                        "(two-endpoint) tests instead.")
        assert model, "Empty model identifier in response."

    def test_both_models_receive_traffic(self, call_endpoint, valid_record):
        """Over many requests, more than one distinct model must appear.

        If only one model ever serves, the A/B split is misconfigured and one
        arm is getting 0% traffic.
        """
        _, body, _ = call_endpoint([valid_record])
        if _model_of(body) is None:
            pytest.skip("Endpoint does not report a 'model' field.")

        seen = {}
        for _ in range(60):
            _, body, _ = call_endpoint([valid_record])
            m = _model_of(body)
            seen[m] = seen.get(m, 0) + 1

        print(f"\n  arms seen: {seen}")
        assert len(seen) >= 2, (
            f"Only one model arm received traffic over 60 requests: {seen}. "
            f"The A/B split is broken -- one arm is getting 0%."
        )

    def test_split_approximately_balanced(self, call_endpoint, valid_record):
        """For a configured ~50/50 split, each arm should land within 35-65%.

        We use a wide tolerance because 60 samples is small; the point is to
        catch a grossly broken split (e.g. 95/5), not to validate exact ratios.
        """
        _, body, _ = call_endpoint([valid_record])
        if _model_of(body) is None:
            pytest.skip("Endpoint does not report a 'model' field.")

        counts = {}
        n = 60
        for _ in range(n):
            _, body, _ = call_endpoint([valid_record])
            m = _model_of(body)
            counts[m] = counts.get(m, 0) + 1

        for arm, c in counts.items():
            share = c / n
            print(f"\n  {arm}: {share:.0%}")
            assert 0.35 <= share <= 0.65, (
                f"Arm {arm} received {share:.0%} of traffic, far from the "
                f"configured 50/50 (expected 35-65%)."
            )


# ===========================================================================
# MODE B — two separate endpoints compared directly
# ===========================================================================

class TestTwoEndpointComparison:
    """Tests comparing ENDPOINT_URL_A and ENDPOINT_URL_B directly."""

    def test_both_endpoints_alive(self, model_a_url, model_b_url, call_named_endpoint, valid_record):
        """Both A and B endpoints must respond 200."""
        sa, _, _ = call_named_endpoint(model_a_url, [valid_record])
        sb, _, _ = call_named_endpoint(model_b_url, [valid_record])
        assert sa == 200, f"Model A endpoint returned HTTP {sa}"
        assert sb == 200, f"Model B endpoint returned HTTP {sb}"

    def test_both_endpoints_return_valid_predictions(
        self, model_a_url, model_b_url, call_named_endpoint, record_bank
    ):
        """Both arms must produce valid in-range predictions on every record."""
        invalid = []
        for case_id, record in record_bank.items():
            _, ba, _ = call_named_endpoint(model_a_url, [record])
            _, bb, _ = call_named_endpoint(model_b_url, [record])
            pa = normalise_prediction(extract_prediction(ba))
            pb = normalise_prediction(extract_prediction(bb))
            if pa not in (0, 1):
                invalid.append(f"A/{case_id}={pa!r}")
            if pb not in (0, 1):
                invalid.append(f"B/{case_id}={pb!r}")
        assert not invalid, f"Invalid predictions: {invalid}"


# ===========================================================================
# Deterministic routing — applies to whichever mode reports a model field
# ===========================================================================

class TestDeterministicRouting:
    """The same input must always route to the same arm."""

    def test_same_input_routes_consistently(self, call_endpoint, valid_record):
        """Submitting the identical record repeatedly must always hit the same
        model arm.

        If a user can be routed to different arms on different requests, they
        could see different predictions for identical input -- the cardinal sin
        of A/B testing.
        """
        _, body, _ = call_endpoint([valid_record])
        first = _model_of(body)
        if first is None:
            pytest.skip("Endpoint does not report a 'model' field; cannot verify routing.")

        arms = set()
        for _ in range(15):
            _, body, _ = call_endpoint([valid_record])
            arms.add(_model_of(body))

        assert arms == {first}, (
            f"The same input routed to multiple arms {arms}. Routing must be "
            f"deterministic (e.g. hash the record id), or a user could get "
            f"different answers for identical input."
        )
