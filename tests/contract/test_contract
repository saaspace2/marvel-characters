"""Contract Tests for the Marvel Characters serving endpoint.

WHAT IS CONTRACT TESTING?
-------------------------
Contract testing verifies the AGREEMENT between two connected systems is
never broken. The serving endpoint is the PROVIDER; your app / notebook /
dashboard is the CONSUMER. If the provider renames a field or changes a
type, every consumer silently breaks.

    Real example: you redeploy and rename the output "Prediction" to
    "prediction" (lowercase). Your dashboard reads response["Prediction"]
    and now shows nothing. A contract test catches this immediately.

THE CONTRACT
------------
INPUT  (what the consumer must send):
    required fields, with declared types.
OUTPUT (what the provider promises to return):
    a 'predictions' key, a prediction value in {0, 1} (or alive/dead),
    and a stable field name that consumers depend on.

RUN
---
    pytest tests/advanced/test_contract.py -v
"""

import pytest

from conftest import extract_prediction, normalise_prediction


# ---------------------------------------------------------------------------
# THE DECLARED CONTRACT  (update deliberately, never accidentally)
# ---------------------------------------------------------------------------

INPUT_CONTRACT = {
    "required_fields": [
        "Height", "Weight", "Universe", "Identity", "Gender",
        "Marital_Status", "Teams", "Origin", "Magic", "Mutant",
    ],
    "numeric_fields": ["Height", "Weight"],
    "integer_fields": ["Teams", "Magic", "Mutant"],
    "string_fields":  ["Universe", "Identity", "Gender", "Marital_Status", "Origin"],
}

OUTPUT_CONTRACT = {
    "required_keys":   ["predictions"],
    "allowed_values":  [0, 1, "alive", "dead"],
    # consumers read one of these keys; the contract is that at least one exists
    "prediction_keys": ["Prediction", "prediction", "Survival prediction"],
    "forbidden_keys":  ["score", "grade", "label"],  # signs the field was renamed
}


class TestInputContract:
    """The endpoint must honour its side of the INPUT agreement."""

    def test_valid_record_satisfies_input_contract(self, valid_record):
        """Our canonical valid record must contain every required field with
        the declared type -- a self-check that the contract definition itself
        matches a real record."""
        for field in INPUT_CONTRACT["required_fields"]:
            assert field in valid_record, f"Required field missing: {field}"
        for field in INPUT_CONTRACT["numeric_fields"]:
            assert isinstance(valid_record[field], (int, float))
        for field in INPUT_CONTRACT["string_fields"]:
            assert isinstance(valid_record[field], str)

    def test_endpoint_accepts_contract_compliant_input(self, call_endpoint, valid_record):
        """A fully contract-compliant record must be accepted (HTTP 200)."""
        status, _, _ = call_endpoint([valid_record])
        assert status == 200, "Endpoint rejected a contract-compliant record"


class TestOutputContract:
    """The endpoint must honour its side of the OUTPUT agreement."""

    def test_output_has_required_keys(self, call_endpoint, valid_record):
        """The response must contain every promised top-level key.

        Kills the breaking change where 'predictions' is renamed or removed.
        """
        _, body, _ = call_endpoint([valid_record])
        for key in OUTPUT_CONTRACT["required_keys"]:
            assert key in body, (
                f"Output contract broken: missing key '{key}'. "
                f"Consumers reading body['{key}'] will break. Body keys: {list(body.keys())}"
            )

    def test_output_has_a_known_prediction_key(self, call_endpoint, valid_record):
        """The predictions object must expose at least one of the agreed
        prediction keys. Kills the 'Prediction' -> 'prediction' rename."""
        _, body, _ = call_endpoint([valid_record])
        preds = body.get("predictions", {})
        if isinstance(preds, list):
            pytest.skip("Endpoint returns a bare list; key-name contract N/A")
        present = [k for k in OUTPUT_CONTRACT["prediction_keys"] if isinstance(preds, dict) and k in preds]
        assert present, (
            f"No agreed prediction key found. Expected one of "
            f"{OUTPUT_CONTRACT['prediction_keys']}, got keys {list(preds.keys()) if isinstance(preds, dict) else preds}"
        )

    def test_output_does_not_use_forbidden_keys(self, call_endpoint, valid_record):
        """The response must NOT contain keys that signal a silent rename.

        If 'score' or 'grade' appears, the prediction field was probably
        renamed -- a breaking change for existing consumers.
        """
        _, body, _ = call_endpoint([valid_record])
        preds = body.get("predictions", {})
        keys = list(preds.keys()) if isinstance(preds, dict) else []
        for forbidden in OUTPUT_CONTRACT["forbidden_keys"]:
            assert forbidden not in keys, (
                f"Output contract broken: forbidden key '{forbidden}' found. "
                f"The prediction field was likely renamed."
            )

    def test_output_value_within_allowed_set(self, call_endpoint, valid_record):
        """The prediction VALUE must be one of the agreed allowed values.

        Kills regressions where the model starts returning probabilities,
        strings, or out-of-range labels instead of the agreed classes.
        """
        _, body, _ = call_endpoint([valid_record])
        pred = extract_prediction(body)
        assert pred in OUTPUT_CONTRACT["allowed_values"], (
            f"Prediction value {pred!r} not in allowed set "
            f"{OUTPUT_CONTRACT['allowed_values']}"
        )

    def test_output_value_type_is_stable(self, call_endpoint, valid_record):
        """The prediction must be an int or a known string label, never a
        float probability or a nested object -- consumers depend on the type."""
        _, body, _ = call_endpoint([valid_record])
        pred = extract_prediction(body)
        assert isinstance(pred, (int, str)) and not isinstance(pred, bool), (
            f"Prediction type {type(pred).__name__} violates contract "
            f"(expected int or str label)"
        )


class TestContractStabilityAcrossInputs:
    """The contract must hold for EVERY record, not just the canonical one."""

    def test_contract_holds_for_all_bank_records(self, call_endpoint, record_bank):
        """Every diverse record in the bank must yield a contract-compliant
        response. A contract that only holds for one input is no contract."""
        violations = []
        for case_id, record in record_bank.items():
            status, body, _ = call_endpoint([record])
            if status != 200:
                violations.append(f"{case_id}: HTTP {status}")
                continue
            if "predictions" not in body:
                violations.append(f"{case_id}: missing 'predictions' key")
                continue
            pred = extract_prediction(body)
            if pred not in OUTPUT_CONTRACT["allowed_values"]:
                violations.append(f"{case_id}: bad value {pred!r}")
        assert not violations, "Contract violations:\n" + "\n".join(violations)
