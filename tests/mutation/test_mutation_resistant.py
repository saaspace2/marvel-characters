"""Mutation-Resistant Unit Tests for the Marvel Characters project.

PURPOSE
-------
These tests are written specifically to KILL MUTANTS.

A normal test checks "does the happy path work?".
A mutation-resistant test also pins down every boundary, sign, branch,
and constant so that if a mutation tool flips an operator or changes a
value, at least one assertion here will fail and "kill" that mutant.

HOW MUTATION TESTING USES THESE TESTS
--------------------------------------
The `mutmut` tool will take the Marvel source code and create mutants like:

    Original:  return confidence < 0.90
    Mutant A:  return confidence > 0.90     (comparison flipped)
    Mutant B:  return confidence <= 0.90    (boundary shifted)
    Mutant C:  return confidence < 0.91     (constant changed)

For each mutant, mutmut re-runs THIS file. If any test below fails, the
mutant is "killed" (good). If all tests still pass, the mutant "survived"
(a gap in our tests that we must close).

EACH TEST BELOW DOCUMENTS WHICH MUTANTS IT IS DESIGNED TO KILL.

HOW TO RUN AS NORMAL UNIT TESTS
--------------------------------
    pytest tests/unit/test_mutation_resistant.py -v

HOW TO RUN UNDER MUTATION TESTING
----------------------------------
    pip install mutmut
    mutmut run
    mutmut results
"""

import math

import numpy as np
import pytest


# ===========================================================================
# Helpers under test
#
# In a real project these live in src/marvel_characters/.  We re-declare
# small reference copies here ONLY where a pure function is simple enough to
# test in isolation.  For functions that already exist in the source package,
# import them directly (shown in the import-style tests further down).
# ===========================================================================


def normalise_marks(raw, max_marks):
    """Clamp a value into the inclusive range [0, max_marks]."""
    return max(0, min(raw, max_marks))


def is_recheck_eligible(confidence):
    """A prediction is eligible for recheck when confidence is BELOW 0.90."""
    return confidence < 0.90


def adjust_predictions_reference(predictions):
    """Convert binary model output (0/1) into human-readable survival labels.

    This mirrors marvel_characters.models.custom_model.adjust_predictions.
    1 -> "alive", 0 -> "dead".
    """
    labels = ["alive" if int(p) == 1 else "dead" for p in predictions]
    return {"Survival prediction": labels}


# ===========================================================================
# SECTION A — BOUNDARY MUTANTS
# Kill mutants that shift comparison boundaries (>= to >, <= to <, etc.)
# ===========================================================================

class TestBoundaryMutants:
    """These tests nail down EXACT boundary values.

    The single most common mutation is shifting a comparison boundary:
        x >= 0   ->   x > 0      (rejects the zero case)
        x <= 10  ->   x < 10     (rejects the max case)

    To kill these mutants you MUST assert the behaviour AT the boundary,
    not just above and below it.
    """

    def test_normalise_at_zero_boundary(self):
        """Killing mutant: min(raw, max) where the lower clamp uses max(0, ...).

        A mutant changing `max(0, ...)` to `max(1, ...)` or the comparison
        direction would let -1 through or push 0 to 1. We pin 0 exactly.
        """
        assert normalise_marks(0, 10) == 0      # exact lower boundary must stay 0
        assert normalise_marks(-0.0, 10) == 0   # negative zero must also be 0

    def test_normalise_at_max_boundary(self):
        """Killing mutant: the upper clamp min(raw, max_marks).

        A mutant changing `min` to `max`, or shifting the boundary, would
        let values above max_marks through. We pin the exact max.
        """
        assert normalise_marks(10, 10) == 10    # exact upper boundary must stay 10
        assert normalise_marks(10.0, 10) == 10  # float at max boundary

    def test_normalise_just_below_zero_is_clamped(self):
        """A value just below zero must be clamped UP to exactly 0.

        Kills mutants that flip the lower-bound comparison.
        """
        assert normalise_marks(-0.001, 10) == 0
        assert normalise_marks(-1, 10) == 0
        assert normalise_marks(-999999, 10) == 0

    def test_normalise_just_above_max_is_clamped(self):
        """A value just above max must be clamped DOWN to exactly max.

        Kills mutants that flip the upper-bound comparison.
        """
        assert normalise_marks(10.001, 10) == 10
        assert normalise_marks(11, 10) == 10
        assert normalise_marks(999999, 10) == 10

    def test_normalise_values_inside_range_unchanged(self):
        """Values strictly inside the range must pass through unchanged.

        Kills mutants that always clamp (e.g. replacing the body with a
        constant) by proving mid-range values are returned as-is.
        """
        assert normalise_marks(5, 10) == 5
        assert normalise_marks(7.5, 10) == 7.5
        assert normalise_marks(0.001, 10) == 0.001
        assert normalise_marks(9.999, 10) == 9.999


# ===========================================================================
# SECTION B — COMPARISON-DIRECTION MUTANTS
# Kill mutants that reverse a comparison (< to >, == to !=)
# ===========================================================================

class TestComparisonDirectionMutants:
    """These tests pin down the DIRECTION of every comparison.

    A mutant flipping `confidence < 0.90` to `confidence > 0.90` reverses
    the entire meaning. To kill it you must assert BOTH a True case and a
    False case so the reversed version cannot satisfy both.
    """

    def test_low_confidence_is_eligible(self):
        """Low confidence -> eligible (True). Kills the flipped `>` mutant."""
        assert is_recheck_eligible(0.50) is True
        assert is_recheck_eligible(0.70) is True
        assert is_recheck_eligible(0.89) is True

    def test_high_confidence_is_not_eligible(self):
        """High confidence -> not eligible (False). Kills the flipped `>` mutant."""
        assert is_recheck_eligible(0.95) is False
        assert is_recheck_eligible(0.99) is False
        assert is_recheck_eligible(1.0) is False

    def test_confidence_exactly_at_threshold_is_not_eligible(self):
        """At exactly 0.90 the result must be False (uses strict `<`).

        This is the killer test for boundary mutants:
            0.90 < 0.90   -> False   (correct)
            0.90 <= 0.90  -> True    (mutant -- caught here)
        """
        assert is_recheck_eligible(0.90) is False

    def test_confidence_just_below_threshold_is_eligible(self):
        """Just below 0.90 must be True. Pins the exact threshold constant.

        Kills mutants that change the constant 0.90 to 0.89 or 0.91:
            0.899 < 0.90  -> True    (correct)
            0.899 < 0.89  -> False   (mutant -- caught here)
        """
        assert is_recheck_eligible(0.899) is True
        assert is_recheck_eligible(0.8999) is True


# ===========================================================================
# SECTION C — VALUE / CONSTANT MUTANTS
# Kill mutants that change a literal value (0 -> 1, "alive" -> "dead")
# ===========================================================================

class TestValueMappingMutants:
    """These tests pin down EXACT output values and string mappings.

    A mutant changing the mapping `1 -> "alive"` to `1 -> "dead"`, or
    swapping the dictionary key, must be caught by asserting the exact
    expected strings and keys.
    """

    def test_one_maps_to_alive_exactly(self):
        """1 must map to the exact string 'alive'.

        Kills mutants that swap the labels or change the string literal.
        """
        result = adjust_predictions_reference([1])
        assert result == {"Survival prediction": ["alive"]}

    def test_zero_maps_to_dead_exactly(self):
        """0 must map to the exact string 'dead'.

        Kills mutants that swap the labels or change the string literal.
        """
        result = adjust_predictions_reference([0])
        assert result == {"Survival prediction": ["dead"]}

    def test_dictionary_key_is_exactly_survival_prediction(self):
        """The output key must be exactly 'Survival prediction'.

        Kills mutants that rename the key (e.g. to 'survival_prediction'
        or 'Prediction'), which would silently break downstream consumers.
        """
        result = adjust_predictions_reference([1])
        assert "Survival prediction" in result
        assert list(result.keys()) == ["Survival prediction"]

    def test_order_is_preserved_exactly(self):
        """Output order must match input order position-by-position.

        Kills mutants that reverse, sort, or reorder the list:
            [1, 0, 1] -> ["alive", "dead", "alive"]   (correct)
            reversed   -> ["alive", "dead", "alive"]   wouldn't differ here,
        so we use an asymmetric input to force order to matter.
        """
        result = adjust_predictions_reference([1, 0, 0])
        assert result == {"Survival prediction": ["alive", "dead", "dead"]}
        # Asymmetric input -- a reversed mutant would give ["dead","dead","alive"]

    def test_mixed_sequence_maps_each_element(self):
        """Each element maps independently. Kills mutants that map only the
        first element or apply one label to the whole list."""
        result = adjust_predictions_reference([1, 0, 1, 0, 1])
        assert result == {
            "Survival prediction": ["alive", "dead", "alive", "dead", "alive"]
        }


# ===========================================================================
# SECTION D — TYPE-HANDLING MUTANTS
# Kill mutants that break numpy/int handling
# ===========================================================================

class TestTypeHandlingMutants:
    """Kill mutants in the int() conversion and numpy handling.

    A mutant removing the `int(p)` cast, or changing `== 1` to `== 0`,
    would break numpy array inputs. These tests use real numpy arrays.
    """

    def test_numpy_array_input_handled(self):
        """numpy int64 values must map correctly, not just python ints.

        Kills mutants that remove the int() cast: np.int64(1) == 1 is True
        in Python, but a mutant changing the comparison would still break.
        """
        result = adjust_predictions_reference(np.array([1, 0]))
        assert result == {"Survival prediction": ["alive", "dead"]}

    def test_numpy_all_ones(self):
        """An all-ones numpy array maps entirely to 'alive'."""
        result = adjust_predictions_reference(np.array([1, 1, 1]))
        assert result == {"Survival prediction": ["alive", "alive", "alive"]}

    def test_numpy_all_zeros(self):
        """An all-zeros numpy array maps entirely to 'dead'."""
        result = adjust_predictions_reference(np.array([0, 0, 0]))
        assert result == {"Survival prediction": ["dead", "dead", "dead"]}


# ===========================================================================
# SECTION E — EMPTY / EDGE MUTANTS
# Kill mutants that mishandle empty input or off-by-one loops
# ===========================================================================

class TestEmptyAndEdgeMutants:
    """Kill mutants that change loop bounds or empty-collection handling.

    A mutant changing a loop's range, or replacing `[]` handling, would
    break on empty input. These tests pin the empty and single-element cases.
    """

    def test_empty_input_returns_empty_list(self):
        """Empty input must return an empty list under the key, not crash.

        Kills mutants that add a default element or change the comprehension.
        """
        result = adjust_predictions_reference([])
        assert result == {"Survival prediction": []}
        assert len(result["Survival prediction"]) == 0

    def test_single_element_produces_single_output(self):
        """One input element must produce exactly one output element.

        Kills off-by-one mutants in loops or comprehensions.
        """
        result = adjust_predictions_reference([1])
        assert len(result["Survival prediction"]) == 1

    def test_output_length_always_matches_input_length(self):
        """Output length must equal input length for any size.

        Kills mutants that drop or duplicate elements.
        """
        for n in (0, 1, 2, 5, 10, 50):
            inp = [1] * n
            result = adjust_predictions_reference(inp)
            assert len(result["Survival prediction"]) == n, (
                f"Input of {n} produced {len(result['Survival prediction'])} outputs"
            )


# ===========================================================================
# SECTION F — IMPORT-STYLE TESTS AGAINST THE REAL SOURCE PACKAGE
# These import the ACTUAL functions from src so mutation testing mutates
# the real code, not the reference copies above.
# ===========================================================================

class TestRealSourceFunctions:
    """Tests that import the real Marvel source functions.

    Mutation testing mutates src/marvel_characters/*.py. To kill those
    mutants, at least some tests must call the REAL functions (not the
    reference copies above). These tests do exactly that.

    If your import paths differ, adjust them to match your package layout.
    """

    def test_real_adjust_predictions_one_to_alive(self):
        """Kill mutants in the real adjust_predictions: 1 -> 'alive'."""
        try:
            from marvel_characters.models.custom_model import adjust_predictions
        except ImportError:
            pytest.skip("marvel_characters.models.custom_model not importable here")

        result = adjust_predictions([1])
        assert result == {"Survival prediction": ["alive"]}

    def test_real_adjust_predictions_zero_to_dead(self):
        """Kill mutants in the real adjust_predictions: 0 -> 'dead'."""
        try:
            from marvel_characters.models.custom_model import adjust_predictions
        except ImportError:
            pytest.skip("marvel_characters.models.custom_model not importable here")

        result = adjust_predictions([0])
        assert result == {"Survival prediction": ["dead"]}

    def test_real_adjust_predictions_empty(self):
        """Kill mutants in the real adjust_predictions: empty -> empty."""
        try:
            from marvel_characters.models.custom_model import adjust_predictions
        except ImportError:
            pytest.skip("marvel_characters.models.custom_model not importable here")

        result = adjust_predictions([])
        assert result == {"Survival prediction": []}

    def test_real_is_databricks_true_when_env_set(self, monkeypatch):
        """Kill mutants in the real is_databricks: env present -> True.

        A mutant flipping the boolean return or the `in os.environ` check
        is caught by asserting both the True and False cases.
        """
        try:
            from marvel_characters.utils import is_databricks
        except ImportError:
            pytest.skip("marvel_characters.utils not importable here")

        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        assert is_databricks() is True

    def test_real_is_databricks_false_when_env_absent(self, monkeypatch):
        """Kill mutants in the real is_databricks: env absent -> False."""
        try:
            from marvel_characters.utils import is_databricks
        except ImportError:
            pytest.skip("marvel_characters.utils not importable here")

        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        assert is_databricks() is False

    def test_real_tags_to_dict_includes_run_id(self):
        """Kill mutants in Tags.to_dict: run_id present -> included."""
        try:
            from marvel_characters.config import Tags
        except ImportError:
            pytest.skip("marvel_characters.config not importable here")

        tags = Tags(git_sha="abc123", branch="main", run_id="run-1")
        result = tags.to_dict()
        assert result == {"git_sha": "abc123", "branch": "main", "run_id": "run-1"}

    def test_real_tags_to_dict_excludes_none_run_id(self):
        """Kill mutants in Tags.to_dict: run_id None -> excluded.

        A mutant flipping the `if run_id is not None` check would include
        a None run_id, which MLflow rejects. This pins the exclusion.
        """
        try:
            from marvel_characters.config import Tags
        except ImportError:
            pytest.skip("marvel_characters.config not importable here")

        tags = Tags(git_sha="abc123", branch="main")
        result = tags.to_dict()
        assert "run_id" not in result
        assert result == {"git_sha": "abc123", "branch": "main"}
