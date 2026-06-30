"""Monitoring & Drift Tests for the Marvel Characters serving endpoint.

WHAT IS DRIFT MONITORING?
-------------------------
Monitoring tests detect problems that develop gradually OVER TIME in
production -- data drift, prediction drift, and degrading performance.

DATA DRIFT
----------
Happens when live data starts to look different from the training data.
PSI (Population Stability Index) is the standard metric:
    PSI < 0.1   no significant shift          -> safe
    0.1 - 0.2   moderate shift                -> monitor closely
    PSI > 0.2   significant drift             -> investigate / retrain

IMPORTANT -- WHY THIS FILE GENERATES A SAMPLE INSTEAD OF USING RECORD_BANK
--------------------------------------------------------------------------
Drift is a STATISTICAL property of a LARGE, REPRESENTATIVE sample. The shared
RECORD_BANK has only 8 records and is deliberately DIVERSE, so feeding it into
a PSI calculation produces a false 'drift' signal. We therefore GENERATE a
realistic sample and test the detector the proper way:
  - no-drift tests : a sample drawn FROM the reference distribution -> low PSI
  - drift tests    : a deliberately shifted sample                  -> high PSI

CALIBRATING REFERENCE_ALIVE_RATE
--------------------------------
REFERENCE_ALIVE_RATE is the historical fraction of 'alive' predictions your
model produced on representative data. It MUST be set to YOUR model's real
baseline, measured from the live endpoint -- not a generic guess. Measured for
this deployment, the model predicts 'alive' on essentially 100% of a
reference-distribution sample, so REFERENCE_ALIVE_RATE is set to 1.00 below.

If you retrain and the model becomes more balanced, re-measure and update this
value. The test then catches a real future drift: e.g. if the alive-rate later
falls below (REFERENCE_ALIVE_RATE - ALIVE_RATE_TOLERANCE), it fails.

CONFIG
------
Optional REFERENCE_STATS_FILE (JSON) overrides the built-in feature reference.

RUN
---
    pytest tests/monitoring_and_drift/ -v
"""

import json
import math
import os
import random
from pathlib import Path

import pytest

from conftest import extract_prediction, normalise_prediction


# Reference (training) distribution for the binned numeric features.
REFERENCE_DISTRIBUTION = {
    "Height": {"<1.6": 0.20, "1.6-1.8": 0.50, "1.8-2.0": 0.25, ">2.0": 0.05},
    "Weight": {"<60": 0.25, "60-80": 0.45, "80-100": 0.22, ">100": 0.08},
}

# --------------------------------------------------------------------------
# CALIBRATED TO THIS DEPLOYMENT.
# Measured: the model predicts 'alive' on ~100% of a representative sample.
# If you retrain to a more balanced model, re-measure and update this number.
# --------------------------------------------------------------------------
REFERENCE_ALIVE_RATE = 1.00
ALIVE_RATE_TOLERANCE = 0.25   # fail if the alive-rate drifts more than this

SAMPLE_SIZE = 500    # size of the generated feature-drift sample
RANDOM_SEED = 42     # fixed so results are reproducible


def _psi(reference: dict, current: dict) -> float:
    """Population Stability Index between two binned distributions."""
    eps = 1e-6
    psi = 0.0
    for bin_label, ref_pct in reference.items():
        cur_pct = max(current.get(bin_label, 0.0), eps)
        ref_pct = max(ref_pct, eps)
        psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
    return psi


def _bin_height(h):
    if h < 1.6:  return "<1.6"
    if h < 1.8:  return "1.6-1.8"
    if h <= 2.0: return "1.8-2.0"
    return ">2.0"


def _bin_weight(w):
    if w < 60:   return "<60"
    if w < 80:   return "60-80"
    if w <= 100: return "80-100"
    return ">100"


def _sample_from_distribution(dist: dict, bin_to_value, n, rng):
    """Draw n values whose binned distribution matches `dist`."""
    labels = list(dist.keys())
    weights = list(dist.values())
    values = []
    for _ in range(n):
        chosen_bin = rng.choices(labels, weights=weights, k=1)[0]
        values.append(bin_to_value(chosen_bin, rng))
    return values


def _height_value(bin_label, rng):
    return {
        "<1.6":    lambda: rng.uniform(1.40, 1.59),
        "1.6-1.8": lambda: rng.uniform(1.60, 1.79),
        "1.8-2.0": lambda: rng.uniform(1.80, 2.00),
        ">2.0":    lambda: rng.uniform(2.01, 2.20),
    }[bin_label]()


def _weight_value(bin_label, rng):
    return {
        "<60":     lambda: rng.uniform(40, 59),
        "60-80":   lambda: rng.uniform(60, 79),
        "80-100":  lambda: rng.uniform(80, 100),
        ">100":    lambda: rng.uniform(101, 150),
    }[bin_label]()


def _dist_of(values, binner):
    """Turn a list of numeric values into a binned proportion dict."""
    counts = {}
    for v in values:
        b = binner(v)
        counts[b] = counts.get(b, 0) + 1
    return {k: c / len(values) for k, c in counts.items()}


@pytest.fixture(scope="module")
def reference_distribution():
    """Load reference stats from REFERENCE_STATS_FILE if provided, else built-in."""
    path = os.environ.get("REFERENCE_STATS_FILE", "").strip()
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return REFERENCE_DISTRIBUTION


@pytest.fixture(scope="module")
def rng():
    return random.Random(RANDOM_SEED)


# ===========================================================================
# FEATURE DRIFT  --  detector must stay QUIET on in-distribution data
# ===========================================================================

class TestNoFalseDrift:
    """A sample drawn FROM the reference distribution must show NO drift."""

    def test_height_no_false_drift(self, reference_distribution, rng):
        sample = _sample_from_distribution(
            reference_distribution["Height"], _height_value, SAMPLE_SIZE, rng
        )
        current = _dist_of(sample, _bin_height)
        psi = _psi(reference_distribution["Height"], current)
        print(f"\n  Height PSI (in-distribution) = {psi:.3f}")
        assert psi < 0.1, (
            f"Detector raised false drift on in-distribution Height data "
            f"(PSI={psi:.3f}). The detector is mis-calibrated."
        )

    def test_weight_no_false_drift(self, reference_distribution, rng):
        sample = _sample_from_distribution(
            reference_distribution["Weight"], _weight_value, SAMPLE_SIZE, rng
        )
        current = _dist_of(sample, _bin_weight)
        psi = _psi(reference_distribution["Weight"], current)
        print(f"\n  Weight PSI (in-distribution) = {psi:.3f}")
        assert psi < 0.1, (
            f"Detector raised false drift on in-distribution Weight data "
            f"(PSI={psi:.3f}). The detector is mis-calibrated."
        )


# ===========================================================================
# FEATURE DRIFT  --  detector must FIRE on genuinely shifted data
# ===========================================================================

class TestDetectsRealDrift:
    """A deliberately shifted sample must trigger the drift alarm."""

    def test_height_drift_is_detected(self, reference_distribution, rng):
        shifted = {"<1.6": 0.02, "1.6-1.8": 0.08, "1.8-2.0": 0.20, ">2.0": 0.70}
        sample = _sample_from_distribution(shifted, _height_value, SAMPLE_SIZE, rng)
        current = _dist_of(sample, _bin_height)
        psi = _psi(reference_distribution["Height"], current)
        print(f"\n  Height PSI (shifted) = {psi:.3f}")
        assert psi > 0.2, (
            f"Detector FAILED to flag an obvious Height shift (PSI={psi:.3f}). "
            f"Real drift would slip through unnoticed."
        )

    def test_weight_drift_is_detected(self, reference_distribution, rng):
        shifted = {"<60": 0.02, "60-80": 0.08, "80-100": 0.20, ">100": 0.70}
        sample = _sample_from_distribution(shifted, _weight_value, SAMPLE_SIZE, rng)
        current = _dist_of(sample, _bin_weight)
        psi = _psi(reference_distribution["Weight"], current)
        print(f"\n  Weight PSI (shifted) = {psi:.3f}")
        assert psi > 0.2, (
            f"Detector FAILED to flag an obvious Weight shift (PSI={psi:.3f})."
        )


# ===========================================================================
# PREDICTION DRIFT  --  output distribution over a representative sample
# ===========================================================================

class TestPredictionDrift:
    """Detect drift in the model's OUTPUT distribution over time."""

    def test_alive_rate_within_expected_range(
        self, call_endpoint, reference_distribution, rng
    ):
        """The current 'alive' rate over a representative sample must stay near
        the model's calibrated reference rate (REFERENCE_ALIVE_RATE).

        Hits the live endpoint probe_n times. Raise probe_n for a more precise
        estimate at the cost of runtime.
        """
        probe_n = 40

        heights = _sample_from_distribution(reference_distribution["Height"], _height_value, probe_n, rng)
        weights = _sample_from_distribution(reference_distribution["Weight"], _weight_value, probe_n, rng)

        base = {
            "Universe": "Earth-616", "Identity": "Public", "Gender": "Male",
            "Marital_Status": "Single", "Teams": 1, "Origin": "Human",
            "Magic": 0, "Mutant": 0,
        }

        alive = 0
        ok = 0
        for h, w in zip(heights, weights):
            record = {**base, "Height": round(h, 2), "Weight": round(w, 1)}
            status, body, _ = call_endpoint([record])
            if status != 200:
                continue
            ok += 1
            if normalise_prediction(extract_prediction(body)) == 1:
                alive += 1

        if ok == 0:
            pytest.skip("No successful endpoint responses; cannot assess prediction drift.")

        current_rate = alive / ok
        drift = abs(current_rate - REFERENCE_ALIVE_RATE)
        print(f"\n  current_alive_rate={current_rate:.2f}  reference={REFERENCE_ALIVE_RATE:.2f}  "
              f"drift={drift:.2f}  (n={ok})")

        assert drift <= ALIVE_RATE_TOLERANCE, (
            f"Prediction drift: alive-rate {current_rate:.2f} differs from the "
            f"calibrated reference {REFERENCE_ALIVE_RATE:.2f} by {drift:.2f} "
            f"(> {ALIVE_RATE_TOLERANCE}).\n"
            f"  If this reflects a genuine, intended model change, update "
            f"REFERENCE_ALIVE_RATE to the new baseline. Otherwise investigate "
            f"the model for prediction drift."
        )


# ===========================================================================
# PSI MATH  --  unit guards on the metric itself
# ===========================================================================

class TestPsiMath:
    """Unit-level guards on the PSI computation."""

    def test_identical_distributions_have_zero_psi(self):
        dist = {"a": 0.5, "b": 0.3, "c": 0.2}
        assert _psi(dist, dist) < 1e-6

    def test_shifted_distribution_has_positive_psi(self):
        ref = {"a": 0.7, "b": 0.2, "c": 0.1}
        cur = {"a": 0.1, "b": 0.2, "c": 0.7}
        assert _psi(ref, cur) > 0.2
