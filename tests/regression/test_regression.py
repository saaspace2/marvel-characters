"""Regression Tests for the Marvel Characters serving endpoint.

WHAT IS REGRESSION TESTING?
---------------------------
Regression testing checks that new changes have not broken things that used
to work. Each new model version is tested against KNOWN-GOOD results ("golden
outputs") captured from a previously-approved version.

    Analogy: a phone OS update adds features but breaks the camera. That is a
    regression. Regression tests catch it before release.

HOW IT WORKS
------------
1. Once, against an approved model, capture predictions for a fixed set of
   records and save them to golden_outputs.json  (run this file with
   --update-golden, or call capture_golden() below).
2. On every new version, predict the same records and compare to the golden
   file. Any difference is FLAGGED for a human to review.

IMPORTANT
---------
A flagged regression does NOT automatically mean a bug -- the new model may
be genuinely better. The test's job is to FORCE A HUMAN DECISION: is this
change an improvement or a regression?

RUN
---
    # First time (against an approved model) -- capture the golden file:
    pytest tests/advanced/test_regression.py --update-golden

    # Every subsequent version:
    pytest tests/advanced/test_regression.py -v
"""

import json
from pathlib import Path

import pytest

from conftest import extract_prediction, normalise_prediction


GOLDEN_FILE = Path(__file__).parent / "golden_outputs.json"


def pytest_addoption(parser):
    """Register the --update-golden flag (also declared in root conftest if shared)."""
    # Guarded so it does not error if already registered elsewhere.
    try:
        parser.addoption(
            "--update-golden", action="store_true", default=False,
            help="Capture current predictions as the new golden baseline.",
        )
    except ValueError:
        pass


@pytest.fixture(scope="session")
def update_golden(request):
    return request.config.getoption("--update-golden", default=False)


@pytest.fixture(scope="session")
def golden_outputs():
    """Load the saved golden predictions, or an empty dict if none exist yet."""
    if GOLDEN_FILE.exists():
        return json.loads(GOLDEN_FILE.read_text())
    return {}


class TestRegression:
    """Compare current predictions against the approved golden baseline."""

    def test_capture_or_compare_golden(
        self, call_endpoint, record_bank, golden_outputs, update_golden
    ):
        """Either capture a fresh golden file (--update-golden) or compare the
        current model's predictions to the saved golden predictions."""
        current = {}
        for case_id, record in record_bank.items():
            status, body, _ = call_endpoint([record])
            assert status == 200, f"{case_id} returned HTTP {status}"
            current[case_id] = normalise_prediction(extract_prediction(body))

        if update_golden:
            GOLDEN_FILE.write_text(json.dumps(current, indent=2))
            pytest.skip(f"Golden baseline captured to {GOLDEN_FILE.name} "
                        f"({len(current)} cases). Re-run without --update-golden to compare.")

        if not golden_outputs:
            pytest.skip("No golden_outputs.json found. Run once with --update-golden "
                        "against an approved model to create the baseline.")

        regressions = []
        for case_id, expected in golden_outputs.items():
            actual = current.get(case_id)
            if actual != expected:
                regressions.append(
                    f"  {case_id}: golden={expected}  ->  now={actual}"
                )

        assert not regressions, (
            f"REGRESSION DETECTED in {len(regressions)} case(s). "
            f"A human must decide if this is a bug or an improvement:\n"
            + "\n".join(regressions)
            + f"\n\nIf the new behaviour is correct, re-baseline with --update-golden."
        )

    def test_all_golden_cases_still_exist(self, record_bank, golden_outputs):
        """Every case in the golden file must still be present in the record
        bank -- otherwise the baseline silently stops protecting that case."""
        if not golden_outputs:
            pytest.skip("No golden baseline to validate.")
        missing = [c for c in golden_outputs if c not in record_bank]
        assert not missing, (
            f"Golden cases no longer in the record bank: {missing}. "
            f"They are no longer being regression-tested."
        )


class TestPredictionStabilityWithinVersion:
    """A single deployed version must be self-consistent across repeated calls."""

    def test_predictions_stable_across_repeated_calls(self, call_endpoint, record_bank):
        """Calling the same records 3 times must give identical results each
        time. Instability here means the regression baseline itself is unreliable."""
        runs = []
        for _ in range(3):
            run = {}
            for case_id, record in record_bank.items():
                _, body, _ = call_endpoint([record])
                run[case_id] = normalise_prediction(extract_prediction(body))
            runs.append(run)

        unstable = [c for c in record_bank if not (runs[0][c] == runs[1][c] == runs[2][c])]
        assert not unstable, (
            f"Predictions unstable across repeated calls for: {unstable}. "
            f"Regression testing requires a deterministic model."
        )
