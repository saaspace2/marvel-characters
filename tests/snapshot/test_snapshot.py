"""Snapshot Tests for the Marvel Characters serving endpoint.

WHAT IS SNAPSHOT TESTING?
-------------------------
Snapshot testing takes a 'photograph' of the response structure at a known-good
state and compares every future response against that photo to detect
unexpected changes -- especially accidental field renames or removals.

    Good for: accidental field renames, format changes, added/removed keys.
    Not for:  intentional changes (then you update the snapshot on purpose),
              or testing prediction VALUES (that's regression testing).

HOW IT WORKS
------------
1. Once, capture the response SHAPE (its keys and value types, NOT the
   specific prediction values) to api_snapshot.json.
2. On every run, compare the current response shape to the snapshot.
3. A mismatch fails the test until a human either fixes the regression or
   deliberately updates the snapshot with --update-snapshot.

RUN
---
    # First time -- capture the snapshot:
    pytest tests/advanced/test_snapshot.py --update-snapshot

    # Every subsequent run:
    pytest tests/advanced/test_snapshot.py -v
"""

import json
from pathlib import Path

import pytest


SNAPSHOT_FILE = Path(__file__).parent / "api_snapshot.json"


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--update-snapshot", action="store_true", default=False,
            help="Capture the current response shape as the new snapshot.",
        )
    except ValueError:
        pass


@pytest.fixture(scope="session")
def update_snapshot(request):
    return request.config.getoption("--update-snapshot", default=False)


def _shape(obj):
    """Recursively reduce a JSON object to its STRUCTURE only.

    Values are replaced by their type name, so we compare shape (keys + types)
    and not the actual prediction values (which legitimately change).
        {"predictions": {"Prediction": 1}}  ->  {"predictions": {"Prediction": "int"}}
    Lists are reduced to a single-element shape describing their item type.
    """
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_shape(obj[0])] if obj else []
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return "str"
    if obj is None:
        return "null"
    return type(obj).__name__


class TestSnapshot:
    """Compare the live response shape against the saved snapshot."""

    def test_response_shape_matches_snapshot(
        self, call_endpoint, valid_record, update_snapshot
    ):
        """The response STRUCTURE must match the saved snapshot exactly.

        Detects: renamed keys, removed keys, added keys, changed value types.
        Ignores: the actual prediction value (handled by regression tests).
        """
        status, body, _ = call_endpoint([valid_record])
        assert status == 200
        current_shape = _shape(body)

        if update_snapshot:
            SNAPSHOT_FILE.write_text(json.dumps(current_shape, indent=2))
            pytest.skip(f"Snapshot captured to {SNAPSHOT_FILE.name}. "
                        f"Re-run without --update-snapshot to compare.")

        if not SNAPSHOT_FILE.exists():
            pytest.skip("No api_snapshot.json found. Run once with "
                        "--update-snapshot to create the baseline.")

        saved_shape = json.loads(SNAPSHOT_FILE.read_text())

        if current_shape != saved_shape:
            # Build a readable diff of the top-level keys.
            cur_keys = set(_flatten_keys(current_shape))
            saved_keys = set(_flatten_keys(saved_shape))
            added   = cur_keys - saved_keys
            removed = saved_keys - cur_keys
            msg = ["Response shape changed from the saved snapshot."]
            if added:
                msg.append(f"  ADDED keys:   {sorted(added)}")
            if removed:
                msg.append(f"  REMOVED keys: {sorted(removed)}")
            if not added and not removed:
                msg.append("  A value TYPE changed (e.g. int -> float).")
            msg.append("If this change is intentional, re-run with --update-snapshot.")
            pytest.fail("\n".join(msg))

    def test_prediction_key_present_in_snapshot(self, call_endpoint, valid_record):
        """Sanity: the live response must still expose the 'predictions' key.

        This is a lighter standalone check that runs even before any snapshot
        file exists, guarding the single most important field.
        """
        _, body, _ = call_endpoint([valid_record])
        assert "predictions" in body, (
            "Critical: 'predictions' key absent from response. "
            "Downstream consumers will break."
        )


def _flatten_keys(shape, prefix=""):
    """Yield dotted key paths from a nested shape dict, for readable diffs."""
    keys = []
    if isinstance(shape, dict):
        for k, v in shape.items():
            path = f"{prefix}.{k}" if prefix else k
            keys.append(path)
            keys.extend(_flatten_keys(v, path))
    elif isinstance(shape, list) and shape:
        keys.extend(_flatten_keys(shape[0], f"{prefix}[]"))
    return keys
