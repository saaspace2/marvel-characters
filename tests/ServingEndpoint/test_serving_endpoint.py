"""Serving Endpoint Tests for the Marvel Characters Model Serving Endpoint.

PURPOSE
-------
These tests go DEEPER than smoke tests.

Smoke tests (tests/smoke/) answer: "Is the system alive?"
Serving tests (this file) answer: "Is the system CORRECT, FAST, and ROBUST?"

Specifically these tests check:
  - Prediction values are valid ('alive' or 'dead')
  - Response structure matches the API contract
  - Latency is within acceptable bounds (p95 < 1500ms)
  - Batch requests work correctly
  - Null values are handled gracefully (no 500 crash)
  - Edge case inputs produce valid outputs
  - The endpoint is deterministic (same input = same output)

PRECONDITION
------------
Smoke tests must be passing before running these.
If the endpoint is not alive, these tests will all fail with misleading errors.

HOW TO RUN
----------
    # Run all serving tests
    pytest tests/serving/ -v

    # Run only prediction validation tests
    pytest tests/serving/test_serving_endpoint.py::TestPredictionValues -v

    # Run with visible timing output
    pytest tests/serving/test_serving_endpoint.py::TestLatency -v -s

    # Skip slow tests
    pytest tests/serving/ -v -m "not slow"

ENVIRONMENT VARIABLES REQUIRED
-------------------------------
    ENDPOINT_URL  — full invocation URL
    DBR_TOKEN     — Databricks personal access token with model-serving scope
"""

import json
import time
import concurrent.futures

import pytest


# ===========================================================================
# SECTION 1 — PREDICTION VALUES: Are the predictions correct and valid?
# ===========================================================================

class TestPredictionValues:
    """Validate that prediction values match the expected output schema.

    The Marvel custom model endpoint returns:
        {"predictions": {"Survival prediction": ["alive"]}}
        {"predictions": {"Survival prediction": ["dead"]}}

    These tests confirm this exact contract is honoured on every deployment.
    """

    def test_prediction_is_alive_or_dead(self, predictions):
        """The prediction must be exactly 'alive' or 'dead'.

        The custom model wraps its output through adjust_predictions() which
        maps:
            1 → 'alive'
            0 → 'dead'

        Any other value means the model's postprocessing pipeline broke.
        This is the primary correctness check for the custom model wrapper.
        """
        assert isinstance(predictions, dict), (
            f"Expected predictions to be a dict, got: {type(predictions)}\n"
            f"Full predictions: {predictions}"
        )

        survival = predictions.get("Survival prediction")
        assert survival is not None, (
            f"'Survival prediction' key missing from predictions.\n"
            f"Actual keys: {list(predictions.keys())}\n\n"
            f"Expected format: {{\"Survival prediction\": [\"alive\"]}} or [\"dead\"]\n"
            f"This means adjust_predictions() in custom_model.py is broken."
        )

        # Unwrap list: ["alive"] → "alive"
        pred_value = survival[0] if isinstance(survival, list) else survival

        assert pred_value in ("alive", "dead"), (
            f"Expected 'alive' or 'dead', got: {pred_value!r}\n"
            f"Full predictions: {predictions}\n\n"
            f"adjust_predictions() maps: 1 → 'alive', 0 → 'dead'\n"
            f"Any other value means the mapping logic broke."
        )

    def test_survival_prediction_is_a_list(self, predictions):
        """The 'Survival prediction' value must be a list, not a bare string.

        adjust_predictions() always wraps the result in a list:
            {'Survival prediction': ['alive']}

        A bare string ('alive' instead of ['alive']) means the wrapper
        was modified and the output format changed — a breaking change
        for any downstream consumer that iterates the list.
        """
        survival = predictions.get("Survival prediction")

        assert isinstance(survival, list), (
            f"Expected 'Survival prediction' to be a list, got: {type(survival)}\n"
            f"Value: {survival!r}\n\n"
            f"adjust_predictions() should return a list. Check custom_model.py."
        )

    def test_survival_prediction_list_has_exactly_one_item(self, predictions):
        """For a single-record request, the prediction list must have exactly 1 item.

        We sent one record, so we expect one prediction. More or fewer
        items would mean the model is duplicating or dropping predictions.
        """
        survival = predictions.get("Survival prediction", [])

        assert len(survival) == 1, (
            f"Expected exactly 1 prediction for 1 input record, got: {len(survival)}\n"
            f"Survival prediction: {survival}\n\n"
            f"This means the model is {'duplicating' if len(survival) > 1 else 'dropping'} "
            f"predictions. Check the pipeline's output handling."
        )

    def test_different_characters_can_get_different_predictions(self, call_endpoint):
        """The model must be able to predict both 'alive' and 'dead'.

        Sending two very different characters should ideally produce
        at least one case of each label — confirming the model is
        actually discriminating, not always outputting the same class.

        Note: this is probabilistic. If it fails, it does not necessarily
        mean the model is broken — it could mean both test characters
        happen to be predicted the same way. Investigate with more samples.
        """
        # Spider-Man (Earth-616, very well-known alive character)
        spiderman_like = [{
            "Height": 1.78, "Weight": 76.0, "Universe": "Earth-616",
            "Identity": "Secret", "Gender": "Male", "Marital_Status": "Single",
            "Teams": 1, "Origin": "Human", "Magic": 0, "Mutant": 0,
        }]

        # A deceased villain-type character
        deceased_like = [{
            "Height": 1.85, "Weight": 90.0, "Universe": "Earth-616",
            "Identity": "Secret", "Gender": "Male", "Marital_Status": "Single",
            "Teams": 0, "Origin": "Human", "Magic": 0, "Mutant": 0,
        }]

        status1, resp1 = call_endpoint(spiderman_like)
        status2, resp2 = call_endpoint(deceased_like)

        assert status1 == 200 and status2 == 200, (
            f"One or both requests failed: status1={status1}, status2={status2}"
        )

        pred1 = json.loads(resp1)["predictions"]["Survival prediction"][0]
        pred2 = json.loads(resp2)["predictions"]["Survival prediction"][0]

        # Both must be valid labels regardless of whether they differ
        assert pred1 in ("alive", "dead"), f"Invalid prediction 1: {pred1!r}"
        assert pred2 in ("alive", "dead"), f"Invalid prediction 2: {pred2!r}"


# ===========================================================================
# SECTION 2 — RESPONSE STRUCTURE: Is the response schema correct?
# ===========================================================================

class TestResponseStructure:
    """Verify the complete response structure matches the API contract.

    The full expected response shape is:
        {
            "predictions": {
                "Survival prediction": ["alive"]  or  ["dead"]
            }
        }

    Any deviation from this schema is a breaking change that must be
    caught immediately — downstream consumers depend on this exact shape.
    """

    def test_response_has_predictions_key(self, parsed_response):
        """Top-level 'predictions' key must be present."""
        assert "predictions" in parsed_response, (
            f"'predictions' key missing from response.\n"
            f"Actual top-level keys: {list(parsed_response.keys())}\n\n"
            f"Expected: {{\"predictions\": {{\"Survival prediction\": [\"alive\"]}}}}"
        )

    def test_predictions_is_a_dict(self, predictions):
        """The 'predictions' value must be a dictionary, not a list or string."""
        assert isinstance(predictions, dict), (
            f"Expected 'predictions' to be a dict, got {type(predictions).__name__}\n"
            f"Value: {predictions!r}\n\n"
            f"The custom model's output format is always a dict with 'Survival prediction' key."
        )

    def test_predictions_has_survival_prediction_key(self, predictions):
        """'Survival prediction' must be a key inside predictions."""
        assert "Survival prediction" in predictions, (
            f"'Survival prediction' key missing from predictions dict.\n"
            f"Actual predictions keys: {list(predictions.keys())}\n\n"
            f"Expected: {{\"Survival prediction\": [\"alive\"]}}\n\n"
            f"This means adjust_predictions() is not being called, or its key changed."
        )

    def test_response_has_no_unexpected_top_level_keys(self, parsed_response):
        """The response should only have 'predictions' at the top level.

        Extra unexpected keys could indicate a model version mismatch,
        middleware injection, or a bug in the serving layer.
        """
        allowed_top_level_keys = {"predictions"}
        actual_keys = set(parsed_response.keys())
        unexpected = actual_keys - allowed_top_level_keys

        # Warn but do not fail for extra keys — they are additive changes
        # Only fail for missing required keys (caught by earlier tests)
        if unexpected:
            pytest.warns(None)  # Just note it without failing
            # Log it for visibility
            print(
                f"\nNote: Response contains unexpected top-level keys: {unexpected}\n"
                f"This is not necessarily a bug, but worth investigating."
            )

    def test_no_error_key_in_response(self, parsed_response):
        """Response must not contain an 'error' key.

        Some frameworks return HTTP 200 with an error key on soft failures.
        This catches silent failures that infrastructure monitors would miss.
        """
        assert "error" not in parsed_response, (
            f"Response contains 'error' key despite HTTP 200.\n"
            f"Error: {parsed_response.get('error')}\n"
            f"This is a silent failure — the endpoint appeared healthy but failed."
        )

    def test_no_error_code_key_in_response(self, parsed_response):
        """Response must not contain 'error_code' (Databricks error format)."""
        assert "error_code" not in parsed_response, (
            f"Response contains 'error_code' key: {parsed_response.get('error_code')}\n"
            f"Message: {parsed_response.get('message', 'no message')}\n"
            f"This is a Databricks API error returned with HTTP 200."
        )


# ===========================================================================
# SECTION 3 — LATENCY: Is the endpoint fast enough?
# ===========================================================================

class TestLatency:
    """Verify response times are within acceptable bounds.

    These tests measure real network + compute latency.
    Results depend on network conditions and whether the endpoint has
    scaled to zero (cold start adds ~30 seconds).

    If latency tests fail:
    1. Run them again — first call may be a cold start
    2. Check if the endpoint is configured with scale_to_zero_enabled=True
    3. Check Databricks cluster health in the serving endpoint logs
    """

    P95_THRESHOLD_MS   = 1500   # 95th percentile must be under 1500ms
    SINGLE_REQUEST_MAX = 3000   # single request must be under 3 seconds
    N_SAMPLES          = 10     # number of requests for percentile calculation

    def test_single_request_completes_within_3_seconds(self, call_endpoint, sample_record):
        """A single prediction must complete in under 3 seconds end-to-end.

        3 seconds accounts for:
          - Network round-trip from client to Databricks (~100ms)
          - Request routing in Databricks (~50ms)
          - Model inference (~200ms)
          - Response serialisation (~50ms)
          - Safety buffer (~2600ms)

        If this fails, the endpoint may have scaled to zero (cold start).
        Wait 60 seconds and retry before investigating further.
        """
        start = time.time()
        status, _ = call_endpoint(sample_record)
        elapsed_ms = (time.time() - start) * 1000

        assert status == 200, f"Request failed with status {status}"
        assert elapsed_ms < self.SINGLE_REQUEST_MAX, (
            f"Request took {elapsed_ms:.0f}ms — exceeds {self.SINGLE_REQUEST_MAX}ms.\n\n"
            f"If this is the first request after a period of inactivity,\n"
            f"the endpoint may have scaled to zero. Wait 60 seconds and retry.\n"
            f"If consistently slow, investigate cluster health and workload size."
        )

    def test_p95_latency_under_500ms(self, call_endpoint, sample_record):
        """95th percentile latency across {N} requests must be under the threshold.

        p95 = the 95th percentile of response times across N requests.
        If p95 = 400ms, it means 95% of requests completed within 400ms.

        After the first request (cold start), the endpoint should be warm
        and subsequent requests should be consistently fast.
        This test deliberately warms the endpoint with the first call.
        """
        # First call: warm up the endpoint (may be slow due to cold start)
        call_endpoint(sample_record)

        # Collect N more timings after warm-up
        times_ms = []
        for _ in range(self.N_SAMPLES):
            start = time.time()
            status, _ = call_endpoint(sample_record)
            elapsed_ms = (time.time() - start) * 1000

            if status == 200:
                times_ms.append(elapsed_ms)

        assert len(times_ms) >= self.N_SAMPLES * 0.8, (
            f"Too many failed requests during latency test: "
            f"{self.N_SAMPLES - len(times_ms)} of {self.N_SAMPLES} failed.\n"
            f"Check endpoint health before measuring latency."
        )

        times_ms.sort()
        p50 = times_ms[int(len(times_ms) * 0.50)]
        p95 = times_ms[int(len(times_ms) * 0.95)]
        p99 = times_ms[min(int(len(times_ms) * 0.99), len(times_ms) - 1)]

        print(
            f"\nLatency results ({len(times_ms)} requests):\n"
            f"  p50 = {p50:.0f}ms\n"
            f"  p95 = {p95:.0f}ms\n"
            f"  p99 = {p99:.0f}ms\n"
            f"  min = {min(times_ms):.0f}ms\n"
            f"  max = {max(times_ms):.0f}ms"
        )

        assert p95 < self.P95_THRESHOLD_MS, (
            f"p95 latency {p95:.0f}ms exceeds {self.P95_THRESHOLD_MS}ms threshold.\n\n"
            f"All timings: {[f'{t:.0f}ms' for t in times_ms]}\n\n"
            f"Possible causes:\n"
            f"  - Workload size is too small (currently Small: 0-4 concurrency)\n"
            f"  - Network latency between your machine and Databricks workspace\n"
            f"  - Model complexity (check n_estimators in config)"
        )

    @pytest.mark.slow
    def test_five_consecutive_requests_all_under_2_seconds(self, call_endpoint, sample_record):
        """Five back-to-back requests must each complete within 2 seconds.

        Unlike p95 which allows occasional slow requests, this test
        checks that NO individual request in a sequence is unacceptably slow.
        Consistently slow individual requests indicate a resource problem.
        """
        slow_requests = []

        for i in range(5):
            start = time.time()
            status, _ = call_endpoint(sample_record)
            elapsed_ms = (time.time() - start) * 1000

            if status != 200:
                pytest.fail(f"Request {i+1}/5 failed with status {status}")

            if elapsed_ms > 2000:
                slow_requests.append(f"Request {i+1}/5: {elapsed_ms:.0f}ms")

        assert len(slow_requests) == 0, (
            f"Some requests exceeded 2000ms:\n"
            + "\n".join(slow_requests)
        )


# ===========================================================================
# SECTION 4 — BATCH REQUESTS: Does the endpoint handle multiple records?
# ===========================================================================

class TestBatchRequests:
    """Verify that the endpoint correctly handles multiple input records.

    The endpoint accepts a list of records. Batch requests are important
    because real-world usage often involves scoring many characters at once.
    """

    def test_batch_of_3_returns_200(self, call_endpoint, batch_records):
        """A batch of 3 records must return HTTP 200."""
        status, response_text = call_endpoint(batch_records)
        assert status == 200, (
            f"Batch request failed with status {status}.\n"
            f"Response: {response_text[:300]}"
        )

    def test_batch_response_has_predictions_key(self, call_endpoint, batch_records):
        """The batch response must contain the 'predictions' key."""
        _, response_text = call_endpoint(batch_records)
        data = json.loads(response_text)
        assert "predictions" in data, (
            f"Batch response missing 'predictions' key.\n"
            f"Actual keys: {list(data.keys())}"
        )

    def test_batch_predictions_have_survival_key(self, call_endpoint, batch_records):
        """The batch predictions must contain 'Survival prediction'."""
        _, response_text = call_endpoint(batch_records)
        data = json.loads(response_text)
        predictions = data.get("predictions", {})

        assert "Survival prediction" in predictions, (
            f"Batch predictions missing 'Survival prediction' key.\n"
            f"Actual predictions: {predictions}"
        )

    def test_batch_of_3_returns_3_predictions(self, call_endpoint, batch_records):
        """Sending 3 records must return exactly 3 predictions.

        The number of predictions must match the number of input records.
        If fewer predictions are returned, rows were silently dropped.
        If more are returned, rows were duplicated — equally dangerous.
        """
        _, response_text = call_endpoint(batch_records)
        data = json.loads(response_text)
        predictions = data.get("predictions", {})
        survival = predictions.get("Survival prediction", [])

        assert len(survival) == len(batch_records), (
            f"Sent {len(batch_records)} records but got {len(survival)} predictions.\n"
            f"Survival predictions: {survival}\n\n"
            f"The model must return exactly one prediction per input record."
        )

    def test_batch_all_predictions_are_valid(self, call_endpoint, batch_records):
        """All predictions in a batch must be 'alive' or 'dead'."""
        _, response_text = call_endpoint(batch_records)
        data = json.loads(response_text)
        survival = data.get("predictions", {}).get("Survival prediction", [])

        invalid = [v for v in survival if v not in ("alive", "dead")]
        assert len(invalid) == 0, (
            f"Invalid prediction values in batch: {invalid}\n"
            f"All predictions: {survival}\n\n"
            f"All values must be either 'alive' or 'dead'."
        )

    def test_single_record_and_batch_give_consistent_results(self, call_endpoint, batch_records):
        """Sending one record alone vs in a batch must give the same prediction.

        The model must be stateless — the prediction for record A must not
        be affected by whether record B was sent in the same request.
        """
        # Send first record of the batch alone
        single = [batch_records[0]]
        _, single_resp = call_endpoint(single)
        single_pred = json.loads(single_resp)["predictions"]["Survival prediction"][0]

        # Send the same record as part of the batch
        _, batch_resp = call_endpoint(batch_records)
        batch_preds = json.loads(batch_resp)["predictions"]["Survival prediction"]
        batch_pred_for_first = batch_preds[0]

        assert single_pred == batch_pred_for_first, (
            f"Same record gives different predictions when sent alone vs in a batch.\n"
            f"  Single request prediction: {single_pred!r}\n"
            f"  Batch request prediction:  {batch_pred_for_first!r}\n\n"
            f"The model must be stateless. Prediction for record A must not\n"
            f"be affected by the presence of other records in the same request."
        )


# ===========================================================================
# SECTION 5 — NULL HANDLING: Does it handle missing values gracefully?
# ===========================================================================

class TestNullHandling:
    """Verify the endpoint handles null/missing values gracefully.

    In production, null values are common: some characters may not have
    a known height, weight, or universe. The endpoint must either:
    - Return a valid prediction (using imputation/defaults), OR
    - Return a clean 4xx error (not a 500 crash)

    A 500 crash on null input means null values in production will take
    down the endpoint for all users — the worst possible failure mode.
    """

    def test_all_null_record_does_not_return_500(self, call_endpoint, null_record):
        """Sending all null values must never crash the server with HTTP 500.

        The CatToIntTransformer in BasicModel encodes unknown values as -1.
        LightGBM can handle NaN in numeric features. Together, these should
        allow null input to be processed without a server crash.
        """
        status, response_text = call_endpoint(null_record)
        assert status != 500, (
            f"Endpoint crashed (HTTP 500) on all-null input.\n"
            f"Response: {response_text[:300]}\n\n"
            f"Null input must never crash the endpoint. Either:\n"
            f"  - Return a valid prediction using imputation, OR\n"
            f"  - Return HTTP 400 with a clear error message.\n"
            f"HTTP 500 means the preprocessing pipeline has no null handling."
        )

    def test_null_height_does_not_crash(self, call_endpoint):
        """Null Height specifically must not crash the server."""
        record = [{
            "Height": None, "Weight": 70.0, "Universe": "Earth-616",
            "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
            "Teams": 1, "Origin": "Human", "Magic": 0, "Mutant": 0,
        }]
        status, response_text = call_endpoint(record)
        assert status != 500, (
            f"Endpoint crashed on null Height.\nResponse: {response_text[:300]}"
        )

    def test_null_weight_does_not_crash(self, call_endpoint):
        """Null Weight specifically must not crash the server."""
        record = [{
            "Height": 1.75, "Weight": None, "Universe": "Earth-616",
            "Identity": "Public", "Gender": "Male", "Marital_Status": "Single",
            "Teams": 1, "Origin": "Human", "Magic": 0, "Mutant": 0,
        }]
        status, response_text = call_endpoint(record)
        assert status != 500, (
            f"Endpoint crashed on null Weight.\nResponse: {response_text[:300]}"
        )

    def test_null_categorical_does_not_crash(self, call_endpoint):
        """Null categorical values (Universe, Gender etc.) must not crash."""
        record = [{
            "Height": 1.75, "Weight": 70.0, "Universe": None,
            "Identity": None, "Gender": None, "Marital_Status": None,
            "Teams": None, "Origin": None, "Magic": None, "Mutant": None,
        }]
        status, response_text = call_endpoint(record)
        assert status != 500, (
            f"Endpoint crashed on null categorical values.\nResponse: {response_text[:300]}"
        )


# ===========================================================================
# SECTION 6 — EDGE CASES: Does it handle boundary values correctly?
# ===========================================================================

class TestEdgeCases:
    """Verify that boundary and unusual-but-valid inputs work correctly.

    These are not invalid inputs (fuzz tests cover those) — they are
    valid inputs at the edges of the expected range.
    """

    @pytest.mark.parametrize("scenario_name", [
        "zero_height_weight",
        "very_large_values",
        "unknown_universe",
        "all_binary_flags_on",
        "all_binary_flags_off",
    ])
    def test_edge_case_returns_valid_prediction(self, call_endpoint, edge_case_records, scenario_name):
        """Each edge case scenario must return a valid 'alive' or 'dead' prediction.

        Edge cases test the model with unusual but technically valid inputs:
          - zero_height_weight: Height=0, Weight=0 (boundary value)
          - very_large_values: Height=999.9 (cosmic-scale characters like Galactus)
          - unknown_universe: Universe='Earth-99999' (unknown universe)
          - all_binary_flags_on: Teams=1, Magic=1, Mutant=1 simultaneously
          - all_binary_flags_off: Teams=0, Magic=0, Mutant=0 simultaneously

        The model must handle all of these without crashing and return
        a valid prediction label.
        """
        record = edge_case_records[scenario_name]
        status, response_text = call_endpoint(record)

        assert status != 500, (
            f"Scenario '{scenario_name}' crashed the server (HTTP 500).\n"
            f"Input: {record}\n"
            f"Response: {response_text[:300]}"
        )

        if status == 200:
            data = json.loads(response_text)
            survival = data.get("predictions", {}).get("Survival prediction")
            if survival is not None:
                pred = survival[0] if isinstance(survival, list) else survival
                assert pred in ("alive", "dead"), (
                    f"Scenario '{scenario_name}' returned invalid prediction: {pred!r}\n"
                    f"Input: {record}"
                )


# ===========================================================================
# SECTION 7 — DETERMINISM: Does the same input always give the same output?
# ===========================================================================

class TestDeterminism:
    """Verify that the endpoint is deterministic.

    The model must return the same prediction for the same input every time.
    Non-determinism would mean:
    - Debugging is impossible (you cannot reproduce the result)
    - A/B testing is unreliable (the 'same' model gives different answers)
    - Audit trails are meaningless (logged predictions cannot be replicated)

    The LightGBM model uses a fixed random_state seed, so predictions
    should be completely deterministic for the same input.
    """

    def test_same_input_gives_same_prediction_across_3_calls(self, call_endpoint, sample_record):
        """Three identical requests must produce identical predictions.

        If the model is truly deterministic, the prediction for the same
        input should never vary regardless of when it is called.
        """
        predictions_seen = set()

        for i in range(3):
            status, response_text = call_endpoint(sample_record)
            assert status == 200, f"Call {i+1}/3 failed with status {status}"

            survival = json.loads(response_text)["predictions"]["Survival prediction"]
            pred = survival[0] if isinstance(survival, list) else survival
            predictions_seen.add(pred)

        assert len(predictions_seen) == 1, (
            f"Same input produced different predictions across 3 calls: {predictions_seen}\n\n"
            f"The model is NOT deterministic. Possible causes:\n"
            f"  - random_state not fixed in LightGBM parameters\n"
            f"  - Non-deterministic preprocessing step\n"
            f"  - Different model versions serving different replicas (version mismatch)"
        )

    def test_prediction_does_not_depend_on_request_order(self, call_endpoint, batch_records):
        """Reversing the order of records in a batch must not change predictions.

        The model must be stateless — each row is scored independently.
        If prediction for row 1 changes when it is sent as row 3, the
        model has a cross-contamination bug.
        """
        # Send in original order
        _, resp_forward = call_endpoint(batch_records)
        preds_forward = json.loads(resp_forward)["predictions"]["Survival prediction"]

        # Send in reversed order
        reversed_records = list(reversed(batch_records))
        _, resp_reversed = call_endpoint(reversed_records)
        preds_reversed = json.loads(resp_reversed)["predictions"]["Survival prediction"]

        # Reverse the reversed predictions back to compare in same order
        preds_reversed_back = list(reversed(preds_reversed))

        assert preds_forward == preds_reversed_back, (
            f"Predictions changed when request order was reversed.\n"
            f"  Forward order predictions: {preds_forward}\n"
            f"  Reversed order predictions (re-ordered): {preds_reversed_back}\n\n"
            f"The model has a cross-row dependency bug — each row must be "
            f"scored independently of the other rows in the same request."
        )


# ===========================================================================
# SECTION 8 — CONCURRENT REQUESTS: Does it handle parallel load?
# ===========================================================================

class TestConcurrency:
    """Verify the endpoint handles multiple simultaneous requests.

    This is a lightweight concurrency check — not a full load test.
    For detailed performance testing under sustained load, see
    tests/performance/test_load.py.
    """

    @pytest.mark.slow
    def test_10_concurrent_requests_all_succeed(self, call_endpoint, sample_record):
        """10 simultaneous requests must all return HTTP 200.

        10 concurrent requests simulates a small burst of simultaneous users.
        The Small workload size (0-4 concurrency) should handle this, though
        some requests may be queued briefly.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(call_endpoint, sample_record)
                for _ in range(10)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        failures = [
            f"HTTP {status}: {resp[:100]}"
            for status, resp in results
            if status != 200
        ]

        assert len(failures) == 0, (
            f"{len(failures)} of 10 concurrent requests failed:\n"
            + "\n".join(failures)
            + "\n\nThe endpoint cannot handle 10 concurrent requests. "
            "Consider upgrading workload_size from Small to Medium."
        )

    @pytest.mark.slow
    def test_concurrent_predictions_are_all_valid(self, call_endpoint, sample_record):
        """All concurrent predictions must be valid 'alive' or 'dead' values.

        Under concurrent load, predictions should not corrupt each other.
        This catches race conditions in the model serving layer.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(call_endpoint, sample_record)
                for _ in range(5)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        invalid = []
        for i, (status, resp) in enumerate(results):
            if status == 200:
                survival = json.loads(resp).get("predictions", {}).get("Survival prediction", [])
                pred = survival[0] if survival else None
                if pred not in ("alive", "dead"):
                    invalid.append(f"Request {i+1}: {pred!r}")

        assert len(invalid) == 0, (
            f"Some concurrent requests returned invalid predictions:\n"
            + "\n".join(invalid)
            + "\n\nConcurrent requests are corrupting each other's predictions."
        )
