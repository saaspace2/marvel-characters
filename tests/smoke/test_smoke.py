"""Smoke Tests for the Marvel Characters Model Serving Endpoint.

PURPOSE
-------
Smoke tests are the very first tests that run after every deployment.
They answer one simple question: "Is the system alive?"

Think of them like plugging in a TV for the first time -- you are not
testing 4K picture quality or Dolby audio yet.  You are just checking
that it turns on.  If it does not turn on, all other tests are pointless.

WHEN TO RUN
-----------
1. Immediately after every deployment (dev → acc → prd).
2. Before running any other test suite (integration, regression, performance).
3. After any infrastructure change (cluster resize, endpoint config change).
4. On a scheduled basis (every 5 minutes) as a liveness probe in production.

WHAT THESE TESTS CHECK
-----------------------
Layer 1 — Is it alive?
    The endpoint must respond with HTTP 200.
    The response body must not be empty.
    The response body must be valid JSON.

Layer 2 — Does it look right?
    The response must contain the 'predictions' key.
    The prediction value must be 'alive' or 'dead'.
    The response must contain a 'Survival prediction' key inside predictions.

Layer 3 — Is it fast enough?
    A single request must complete in under 3 seconds.
    Five back-to-back requests must ALL succeed (no intermittent failures).

Layer 4 — Does it handle both models?
    Both the basic and custom model endpoints must be alive.

HOW THESE DIFFER FROM OTHER TESTS
-----------------------------------
| Test type    | What it checks          | Speed    | Scope      |
|--------------|-------------------------|----------|------------|
| Smoke        | Is it alive?            | < 5s     | Minimal    |
| Serving      | Are predictions valid?  | ~30s     | Deep       |
| Integration  | Does the pipeline work? | ~2 min   | End-to-end |
| Performance  | Can it handle load?     | ~5 min   | Stress     |

Run smoke first -- if these fail, skip everything else.

HOW TO RUN
----------
    # Run all smoke tests
    pytest tests/smoke/ -v

    # Run only the liveness check (fastest possible check)
    pytest tests/smoke/test_smoke.py::TestLiveness::test_endpoint_returns_200 -v

    # Run with visible print output (useful for debugging)
    pytest tests/smoke/ -v -s
"""

import json
import os
import time

import pytest
import requests


# ===========================================================================
# LAYER 1 — LIVENESS: Is the endpoint alive at all?
# ===========================================================================

class TestLiveness:
    """The most basic checks.  If any of these fail, the deployment failed.
    Do not investigate model quality until these pass.
    """

    def test_endpoint_returns_200(self, live_response):
        """HTTP 200 means the endpoint accepted the request and responded.

        Any other status code means:
          404 → endpoint was not deployed or the URL is wrong
          401 → authentication failed (bad or missing token)
          500 → the model crashed immediately on a valid input
          503 → the endpoint is starting up or overloaded

        This is the single most important smoke test.  If this fails,
        the deployment has failed and must be rolled back before any
        users are affected.
        """
        status, response_text = live_response
        assert status == 200, (
            f"Endpoint returned HTTP {status} instead of 200.\n"
            f"Response body: {response_text[:500]}\n\n"
            f"Common causes:\n"
            f"  404 → ENDPOINT_URL is wrong or endpoint was not deployed\n"
            f"  401 → DBR_TOKEN is expired or does not have permission\n"
            f"  500 → Model crashed on the sample input — check model logs\n"
            f"  503 → Endpoint is still starting up — wait 60s and retry"
        )

    def test_response_body_is_not_empty(self, live_response):
        """A non-empty body confirms the server actually sent something back.

        An empty body with a 200 status would indicate a misconfigured
        endpoint that accepts requests but produces no output, which is
        a silent failure more dangerous than an explicit error.
        """
        _, response_text = live_response
        assert len(response_text) > 0, (
            "Endpoint returned HTTP 200 but the response body is completely empty. "
            "This indicates the model or server is misconfigured."
        )

    def test_response_is_valid_json(self, live_response):
        """The response body must be parseable JSON.

        Databricks Model Serving always returns JSON.  If we get back
        HTML (e.g. a login page or an error page) it means something
        is wrong at the infrastructure level, not the model level.
        """
        _, response_text = live_response
        try:
            json.loads(response_text)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Response body is not valid JSON.\n"
                f"JSON error: {exc}\n"
                f"First 500 chars of response: {response_text[:500]}\n\n"
                f"Common causes:\n"
                f"  - The endpoint returned an HTML error page\n"
                f"  - A proxy/firewall intercepted the request\n"
                f"  - The model returned raw text instead of a JSON payload"
            )


# ===========================================================================
# LAYER 2 — RESPONSE SHAPE: Does the response look correct?
# ===========================================================================

class TestResponseShape:
    """Check that the response contains the expected keys and value types.

    The Marvel custom model endpoint returns:
        {"predictions": {"Survival prediction": ["alive"]}}
        {"predictions": {"Survival prediction": ["dead"]}}

    These tests verify this exact contract is intact after every deployment.
    If a key is missing or renamed, downstream consumers will silently break.
    """

    def test_predictions_key_exists(self, live_response):
        """The top-level 'predictions' key must be present.

        This is the contract between the serving endpoint and every
        consumer of it.  If 'predictions' is missing, the model may have
        been redeployed with a different output schema -- a breaking change
        that must be caught immediately.
        """
        _, response_text = live_response
        data = json.loads(response_text)
        assert "predictions" in data, (
            f"'predictions' key missing from response.\n"
            f"Actual response keys: {list(data.keys())}\n"
            f"Full response: {response_text[:500]}\n\n"
            f"This means the model output schema changed. "
            f"Check whether the model was re-registered with a different signature."
        )

    def test_prediction_value_is_valid(self, live_response):
        """The predicted value must be 'alive' or 'dead'.

        The Marvel character survival classifier wraps its output through
        adjust_predictions() which converts 0/1 integers into human-readable
        labels:  0 → 'dead',  1 → 'alive'.

        The actual response format is:
            {"predictions": {"Survival prediction": ["alive"]}}

        Any value outside {'alive', 'dead'} means the model's postprocessing
        pipeline (adjust_predictions) broke or the model was re-deployed
        without the custom wrapper.
        """
        _, response_text = live_response
        data = json.loads(response_text)

        if "predictions" not in data:
            pytest.skip("Skipping: 'predictions' key not present (caught by earlier test)")

        predictions = data["predictions"]

        # --- Our model returns: {"Survival prediction": ["alive"]} ---
        if isinstance(predictions, dict) and "Survival prediction" in predictions:
            survival = predictions["Survival prediction"]
            # unwrap list if needed: ["alive"] → "alive"
            pred_value = survival[0] if isinstance(survival, list) else survival
            assert pred_value in ("alive", "dead"), (
                f"Expected 'alive' or 'dead', got: {pred_value!r}\n"
                f"Full predictions: {predictions}\n\n"
                f"This means adjust_predictions() returned an unexpected label. "
                f"Check custom_model.py for the mapping logic."
            )
            return

        # --- Fallback: handle binary integer format {"Prediction": 1} ---
        if isinstance(predictions, dict):
            pred_value = predictions.get("Prediction", predictions.get("prediction"))
        elif isinstance(predictions, list):
            pred_value = predictions[0] if predictions else None
        else:
            pred_value = predictions

        assert pred_value in (0, 1, "alive", "dead"), (
            f"Expected prediction value of 0, 1, 'alive' or 'dead', got: {pred_value!r}\n"
            f"Full predictions: {predictions}\n\n"
            f"This means the model is producing invalid output. "
            f"Check the model's classification layer and postprocessing."
        )

    def test_survival_prediction_key_exists(self, live_response):
        """The 'Survival prediction' key must exist inside predictions.

        The custom model wrapper (MarvelModelWrapper via adjust_predictions)
        always returns:
            {"predictions": {"Survival prediction": ["alive or dead"]}}

        If this key is missing it means one of:
          - The model was re-deployed without the custom pyfunc wrapper
          - adjust_predictions() was modified and no longer uses this key
          - The basic model endpoint URL was used instead of the custom one

        This test replaces the previous 'model field' check since the
        custom model endpoint does not return a separate 'model' field --
        it returns 'Survival prediction' as the output key instead.
        """
        _, response_text = live_response
        data = json.loads(response_text)

        if "predictions" not in data:
            pytest.skip("Skipping: 'predictions' key not present (caught by earlier test)")

        predictions = data["predictions"]

        assert isinstance(predictions, dict) and "Survival prediction" in predictions, (
            f"'Survival prediction' key missing from predictions.\n"
            f"Actual predictions: {predictions}\n\n"
            f"Expected format: {{\"Survival prediction\": [\"alive\"]}} or [\"dead\"]\n\n"
            f"This means the custom model wrapper (adjust_predictions) is not "
            f"being applied. Check that the custom pyfunc model was deployed, "
            f"not the raw basic sklearn model."
        )

    def test_response_does_not_contain_error_key(self, live_response):
        """The response must not contain an 'error' key.

        Some serving frameworks return HTTP 200 but include an 'error'
        field in the JSON body when a soft error occurs.  This is a
        silent failure that looks like a success to infrastructure monitors
        but actually means the prediction failed.
        """
        _, response_text = live_response
        data = json.loads(response_text)
        assert "error" not in data, (
            f"Response contained an 'error' key despite returning HTTP 200.\n"
            f"Error value: {data.get('error')}\n"
            f"This is a silent failure -- the endpoint appeared healthy "
            f"but actually failed to produce a valid prediction."
        )


# ===========================================================================
# LAYER 3 — LATENCY: Is it fast enough for real-time use?
# ===========================================================================

class TestLatency:
    """Verify the endpoint responds within acceptable time limits.

    Smoke latency tests use a single request and a tight threshold.
    For detailed percentile analysis (p50, p95, p99) under concurrent
    load, see the performance test suite.
    """

    def test_single_request_under_3_seconds(self, call_endpoint, sample_record):
        """A single prediction must complete in under 3 seconds end-to-end.

        This threshold accounts for:
          - Network round-trip (~50ms)
          - Model inference (~200ms)
          - adjust_predictions() postprocessing (~10ms)
          - Safety margin (~2740ms)

        If this fails consistently, investigate:
          - Cold-start time (endpoint may have scaled to zero)
          - Network latency between client and Databricks workspace
          - Model complexity (n_estimators, max_depth settings)
        """
        start = time.time()
        status, _ = call_endpoint(sample_record)
        elapsed_ms = (time.time() - start) * 1000

        assert status == 200, f"Request failed with status {status}"
        assert elapsed_ms < 3000, (
            f"Request took {elapsed_ms:.0f}ms — exceeds the 3000ms smoke threshold.\n\n"
            f"This does not mean the model is permanently slow; it could be a\n"
            f"cold-start issue if the cluster was idle.  Run again to confirm.\n"
            f"If it consistently exceeds 3000ms, investigate:\n"
            f"  - Cluster warm-up time (consider enabling always-on compute)\n"
            f"  - Network latency to the Databricks workspace\n"
            f"  - Model hyperparameters (n_estimators may be set too high)"
        )

    def test_five_consecutive_requests_all_succeed(self, call_endpoint, sample_record):
        """Five back-to-back requests must ALL return HTTP 200.

        A single success could be a fluke.  Five consecutive successes
        confirm the endpoint is stably deployed, not intermittently failing.
        This is especially important to check after a rolling deployment
        where some replicas may still be starting up.
        """
        failures = []
        for i in range(5):
            status, response_text = call_endpoint(sample_record)
            if status != 200:
                failures.append(
                    f"Request {i + 1}/5 failed: HTTP {status} — {response_text[:100]}"
                )

        assert len(failures) == 0, (
            f"{len(failures)} of 5 consecutive requests failed:\n"
            + "\n".join(failures)
            + "\n\nIntermittent failures indicate an unstable deployment. "
            "Check cluster health and rolling-update status."
        )


# ===========================================================================
# LAYER 4 — BOTH MODELS: Are basic AND custom endpoints alive?
# ===========================================================================

class TestBothModels:
    """Verify that BOTH the basic and custom model endpoints are responding.

    The Marvel pipeline registers two models:
      - marvel_character_model_basic  (sklearn pipeline, returns 0/1)
      - marvel_character_model_custom (pyfunc wrapper, returns {"Survival prediction": [...]})

    Both must be deployed and alive after every release.

    Set these environment variables to enable these tests:
        export ENDPOINT_URL_BASIC="https://.../marvel-character-model-serving/invocations"
        export ENDPOINT_URL_CUSTOM="https://.../marvel-character-model-serving-custom/invocations"
    """

    BASIC_ENDPOINT_ENV  = "ENDPOINT_URL_BASIC"
    CUSTOM_ENDPOINT_ENV = "ENDPOINT_URL_CUSTOM"

    def _call(self, url: str, token: str, record: list[dict]) -> tuple[int, str]:
        """Helper: make one POST request and return (status, text)."""
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"dataframe_records": record},
            timeout=30,
        )
        return response.status_code, response.text

    def test_basic_model_endpoint_is_alive(self, dbr_token, sample_record):
        """The basic sklearn model endpoint must return HTTP 200.

        The basic model produces binary integer predictions (0 or 1).
        If this endpoint is down while the custom one is up, it means
        a partial deployment failure -- only one of the two models was
        successfully served.
        """
        url = os.environ.get(self.BASIC_ENDPOINT_ENV, "").strip()
        if not url:
            pytest.skip(
                f"{self.BASIC_ENDPOINT_ENV} not set — skipping basic model smoke test. "
                f"Set it to test both models independently."
            )

        status, response_text = self._call(url, dbr_token, sample_record)
        assert status == 200, (
            f"Basic model endpoint returned HTTP {status}.\n"
            f"URL: {url}\n"
            f"Response: {response_text[:300]}"
        )

    def test_custom_model_endpoint_is_alive(self, dbr_token, sample_record):
        """The custom pyfunc model endpoint must return HTTP 200.

        The custom model wraps predictions through adjust_predictions()
        and returns a dict with key 'Survival prediction'.
        If this endpoint is down while the basic one is up, it means
        the custom model wrapper failed to deploy correctly.
        """
        url = os.environ.get(self.CUSTOM_ENDPOINT_ENV, "").strip()
        if not url:
            pytest.skip(
                f"{self.CUSTOM_ENDPOINT_ENV} not set — skipping custom model smoke test. "
                f"Set it to test both models independently."
            )

        status, response_text = self._call(url, dbr_token, sample_record)
        assert status == 200, (
            f"Custom model endpoint returned HTTP {status}.\n"
            f"URL: {url}\n"
            f"Response: {response_text[:300]}"
        )

    def test_custom_model_returns_survival_prediction_key(self, dbr_token, sample_record):
        """The custom model response must contain 'Survival prediction'.

        The custom model's adjust_predictions() function converts binary
        integers into human-readable labels inside a dict with this exact
        key.  If the key is missing, the wrapper function broke during
        deployment.
        """
        url = os.environ.get(self.CUSTOM_ENDPOINT_ENV, "").strip()
        if not url:
            pytest.skip(f"{self.CUSTOM_ENDPOINT_ENV} not set — skipping.")

        status, response_text = self._call(url, dbr_token, sample_record)
        if status != 200:
            pytest.skip(f"Custom endpoint returned {status} — caught by liveness test.")

        data = json.loads(response_text)
        predictions = data.get("predictions", {})

        has_key = (
            (isinstance(predictions, dict) and "Survival prediction" in predictions)
            or (hasattr(predictions, "columns") and "Survival prediction" in predictions.columns)
        )
        assert has_key, (
            f"Custom model response missing 'Survival prediction' key.\n"
            f"Actual predictions: {predictions}\n\n"
            f"This means adjust_predictions() in custom_model.py is not being "
            f"called correctly, or the model was re-registered without the wrapper."
        )


# ===========================================================================
# LAYER 5 — GRACEFUL DEGRADATION: Does it fail safely?
# ===========================================================================

class TestGracefulDegradation:
    """Confirm the endpoint handles bad input gracefully instead of crashing.

    A smoke-level degradation test is not the same as fuzz testing.
    We are not throwing hundreds of random inputs -- we are checking
    one or two basic failure modes to confirm the server does not
    crash (HTTP 500) on obviously wrong input.
    """

    def test_empty_payload_does_not_cause_500(self, call_endpoint):
        """Sending an empty list must return 400 (bad request), not 500 (crash).

        400 = the server understood the request and rejected it cleanly.
        500 = the server crashed — the model has no input validation.

        A server crash on empty input means the model will also crash on
        corrupted real-world data, which is far more dangerous in production.
        """
        status, response_text = call_endpoint([])
        assert status != 500, (
            f"Endpoint crashed (HTTP 500) on an empty payload.\n"
            f"Response: {response_text[:300]}\n\n"
            f"The endpoint must return HTTP 400 for invalid input, never 500. "
            f"Add input validation to the model serving handler."
        )

    def test_wrong_field_types_do_not_cause_500(self, call_endpoint):
        """Sending strings where numbers are expected must return 400, not 500.

        This simulates a client sending malformed data.  The endpoint must
        reject it cleanly with a 4xx response rather than crashing with 500.
        """
        bad_record = [
            {
                "Height":         "tall",     # should be float
                "Weight":         "heavy",    # should be float
                "Universe":       "Earth-616",
                "Identity":       "Public",
                "Gender":         "Male",
                "Marital_Status": "Single",
                "Teams":          "yes",      # should be int
                "Origin":         "Human",
                "Magic":          "no",       # should be int
                "Mutant":         "no",       # should be int
            }
        ]
        status, response_text = call_endpoint(bad_record)
        assert status != 500, (
            f"Endpoint crashed (HTTP 500) on wrong field types.\n"
            f"Response: {response_text[:300]}\n\n"
            f"The endpoint must validate field types and return 400, not crash. "
            f"This indicates missing schema enforcement in the serving layer."
        )

    def test_missing_required_fields_do_not_cause_500(self, call_endpoint):
        """Sending a record with missing fields must not crash the server.

        This is the most common real-world error: a client sends a record
        where some fields were null or not collected.  The server must
        handle this gracefully with a 4xx error, not a 500 crash.
        """
        incomplete_record = [{"Height": 1.75, "Weight": 70.0}]  # missing 8 fields
        status, response_text = call_endpoint(incomplete_record)
        assert status != 500, (
            f"Endpoint crashed (HTTP 500) on a record with missing fields.\n"
            f"Response: {response_text[:300]}\n\n"
            f"The endpoint must return 400 for incomplete input, not crash. "
            f"Add null checks or default values in the preprocessing pipeline."
        )
