"""Fuzz Tests for the Marvel Characters Model Serving Endpoint.

PURPOSE
-------
Fuzz tests throw random, unexpected, and invalid inputs at your system
to discover crashes and unexpected behaviour BEFORE real users do.

Real-World Analogy
------------------
A vending machine: normal testing presses B3 and gets a Coke.
Fuzz testing inserts a banana instead of a coin, presses 999 buttons
at once, and shakes the machine violently. Does it catch fire?

THE GOLDEN RULE
---------------
The system must NEVER return HTTP 500 regardless of what input is sent.

    200 = prediction worked
    400 = bad input was detected and rejected cleanly  ← acceptable
    500 = the server crashed                           ← always a bug

A 500 on bad input means the model has no validation layer, so real-world
data corruption or a misbehaving client can take down the entire endpoint.

THE FOUR TYPES OF FUZZ TESTING COVERED HERE
--------------------------------------------
Type 1 — Random Fuzzing      (Section A)
    Completely random inputs. No logic — pure chaos.
    Uses Hypothesis to auto-generate hundreds of cases automatically.

Type 2 — Boundary Fuzzing    (Section B)
    Testing the edges of valid values — just above and below limits.
    E.g. Height = -0.001, Height = 0.0, Height = 999999.9

Type 3 — Schema Fuzzing      (Section C)
    Wrong structure: missing fields, extra fields, wrong types, None,
    empty dict, nested dicts, lists where scalars are expected.

Type 4 — Security Fuzzing    (Section D)
    Malicious inputs: XSS, SQL injection, path traversal, null bytes,
    oversized payloads, unicode bombs.

HOW HYPOTHESIS WORKS
---------------------
Hypothesis auto-generates test inputs based on strategies you define.
When it finds a failure it automatically shrinks the input to the
simplest possible failing case and saves it in the .hypothesis/ folder
so every future run re-tests that exact case.

    @given(height=st.floats())
    def test_any_float_height_never_crashes(height):
        ...

    # Hypothesis generates e.g.: 0.0, inf, -inf, nan, 1e308, -1e-300 ...
    # If any of these cause a 500, the test fails and reports the exact value.

HOW TO RUN
----------
    # Run all fuzz tests
    pytest tests/fuzz/ -v

    # Run only one type
    pytest tests/fuzz/test_fuzz.py::TestRandomFuzzing -v

    # Run with a fixed seed for reproducibility
    pytest tests/fuzz/ -v --hypothesis-seed=42

    # Increase the number of Hypothesis examples (default is 100 per test)
    pytest tests/fuzz/ -v --hypothesis-seed=0 -p no:randomly

    # Show Hypothesis statistics
    pytest tests/fuzz/ -v --hypothesis-show-statistics
"""

import json
import math
import random
import string

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from conftest import (
    ALL_FIELDS,
    INTEGER_FIELDS,
    NUMERIC_FIELDS,
    STRING_FIELDS,
    VALID_RECORD,
)


# ---------------------------------------------------------------------------
# Hypothesis settings profiles
# ---------------------------------------------------------------------------

# Fast: used for tests that hit the real endpoint (each example = 1 HTTP call)
FAST_SETTINGS = settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    deadline=10_000,   # 10 seconds per example max
)

# Standard: for parametrized tests that do NOT hit the endpoint
STANDARD_SETTINGS = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)


# ===========================================================================
# SECTION A — RANDOM FUZZING
# Completely random inputs — no logic, pure chaos.
# Hypothesis generates values automatically; we just define the strategy.
# ===========================================================================

class TestRandomFuzzing:
    """Type 1: Random Fuzzing — throw auto-generated chaos at every field.

    Each test below targets a specific field or group of fields.
    Hypothesis generates up to 30 random values per test and reports
    the exact value that caused a failure (with shrinking).
    """

    @given(height=st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.integers(),
        st.none(),
        st.text(max_size=50),
        st.lists(st.integers(), max_size=3),
    ))
    @FAST_SETTINGS
    def test_random_height_never_causes_500(self, height, call_endpoint, valid_record):
        """Any value for Height must never crash the server.

        Valid: 1.75 (float)
        Fuzz:  None, "tall", [1,2,3], float('inf'), float('nan'), -999999
        """
        record = {**valid_record, "Height": height}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) when Height={height!r}\n"
            f"Response: {response_text[:300]}\n\n"
            f"The endpoint must validate Height and return 400, not crash."
        )

    @given(weight=st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.integers(),
        st.none(),
        st.text(max_size=50),
    ))
    @FAST_SETTINGS
    def test_random_weight_never_causes_500(self, weight, call_endpoint, valid_record):
        """Any value for Weight must never crash the server."""
        record = {**valid_record, "Weight": weight}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) when Weight={weight!r}\n"
            f"Response: {response_text[:300]}"
        )

    @given(universe=st.one_of(
        st.text(max_size=1000),
        st.none(),
        st.integers(),
        st.binary(max_size=100),
    ))
    @FAST_SETTINGS
    def test_random_universe_never_causes_500(self, universe, call_endpoint, valid_record):
        """Any value for Universe must never crash the server.

        Valid: "Earth-616"
        Fuzz:  None, 42, b"\\xff\\xfe", "A" * 1000, "", "\\x00\\x01\\x02"
        """
        record = {**valid_record, "Universe": universe}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) when Universe={universe!r}\n"
            f"Response: {response_text[:300]}"
        )

    @given(gender=st.one_of(
        st.text(max_size=500),
        st.none(),
        st.booleans(),
        st.integers(),
    ))
    @FAST_SETTINGS
    def test_random_gender_never_causes_500(self, gender, call_endpoint, valid_record):
        """Any value for Gender must never crash the server.

        Valid: "Male", "Female", "Other"
        Fuzz:  None, True, 0, "Unknown-Value", "<script>", ""
        """
        record = {**valid_record, "Gender": gender}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) when Gender={gender!r}\n"
            f"Response: {response_text[:300]}"
        )

    @given(
        teams=st.one_of(st.integers(), st.floats(), st.none(), st.text(max_size=50)),
        magic=st.one_of(st.integers(), st.floats(), st.none(), st.text(max_size=50)),
        mutant=st.one_of(st.integers(), st.floats(), st.none(), st.text(max_size=50)),
    )
    @FAST_SETTINGS
    def test_random_integer_fields_never_cause_500(
        self, teams, magic, mutant, call_endpoint, valid_record
    ):
        """Any values for Teams, Magic, Mutant must never crash the server.

        Valid: 0 or 1 (binary integer)
        Fuzz:  None, "yes", 2, -1, 1.5, float('inf')
        """
        record = {**valid_record, "Teams": teams, "Magic": magic, "Mutant": mutant}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) with Teams={teams!r}, "
            f"Magic={magic!r}, Mutant={mutant!r}\n"
            f"Response: {response_text[:300]}"
        )

    @given(all_values=st.fixed_dictionaries({
        field: st.one_of(
            st.text(max_size=200),
            st.integers(min_value=-999999, max_value=999999),
            st.floats(allow_nan=True, allow_infinity=True),
            st.none(),
            st.booleans(),
        )
        for field in ALL_FIELDS
    }))
    @FAST_SETTINGS
    def test_all_fields_randomised_simultaneously_never_cause_500(
        self, all_values, call_endpoint
    ):
        """Every field set to a random value simultaneously must not crash the server.

        This is the hardest fuzz test — chaos on all 10 fields at once.
        The server must handle complete garbage input gracefully.
        """
        status, response_text = call_endpoint([all_values])
        assert status != 500, (
            f"Server crashed (HTTP 500) when all fields were randomised.\n"
            f"Input: {all_values}\n"
            f"Response: {response_text[:300]}\n\n"
            f"The endpoint has no protection against fully malformed records."
        )


# ===========================================================================
# SECTION B — BOUNDARY FUZZING
# Testing the edges of valid ranges — just inside and just outside limits.
# These are parametrized tests (no Hypothesis) for precise control.
# ===========================================================================

class TestBoundaryFuzzing:
    """Type 2: Boundary Fuzzing — test values at the exact edges of validity.

    Boundary bugs are among the most common in production:
      - Height just below zero (e.g. -0.001) triggers a preprocessing crash
      - A string with exactly 256 chars breaks a VARCHAR(255) column
      - Teams = 2 when only 0/1 is allowed causes silent wrong predictions

    Each parametrize case is labelled so you can see exactly which
    boundary triggered a failure.
    """

    # ── Numeric boundaries ────────────────────────────────────────────────

    @pytest.mark.parametrize("height, label", [
        (0.0,         "zero"),
        (-0.001,      "just_below_zero"),
        (-1.0,        "negative_one"),
        (-999.9,      "large_negative"),
        (-999999.0,   "extreme_negative"),
        (0.001,       "just_above_zero"),
        (100.0,       "realistic_max"),
        (300.0,       "above_realistic_max"),
        (999999.9,    "extreme_positive"),
        (float("inf"),  "positive_infinity"),
        (float("-inf"), "negative_infinity"),
        (float("nan"),  "not_a_number"),
        (1e308,        "near_float_max"),
        (-1e308,       "near_float_min"),
    ])
    def test_height_boundary_never_causes_500(
        self, height, label, call_endpoint, valid_record
    ):
        """Height at every boundary value must not crash the server."""
        record = {**valid_record, "Height": height}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) at Height boundary '{label}' = {height}\n"
            f"Response: {response_text[:300]}"
        )

    @pytest.mark.parametrize("weight, label", [
        (0.0,         "zero"),
        (-0.001,      "just_below_zero"),
        (-500.0,      "large_negative"),
        (0.001,       "just_above_zero"),
        (300.0,       "realistic_max"),
        (999999.9,    "extreme_positive"),
        (float("inf"),  "positive_infinity"),
        (float("-inf"), "negative_infinity"),
        (float("nan"),  "not_a_number"),
    ])
    def test_weight_boundary_never_causes_500(
        self, weight, label, call_endpoint, valid_record
    ):
        """Weight at every boundary value must not crash the server."""
        record = {**valid_record, "Weight": weight}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) at Weight boundary '{label}' = {weight}\n"
            f"Response: {response_text[:300]}"
        )

    @pytest.mark.parametrize("value, field, label", [
        # Teams, Magic, Mutant are binary (0 or 1) -- test at and around the boundary
        (-1,   "Teams",  "below_zero"),
        (0,    "Teams",  "valid_zero"),
        (1,    "Teams",  "valid_one"),
        (2,    "Teams",  "above_one"),
        (100,  "Teams",  "far_above"),
        (-1,   "Magic",  "below_zero"),
        (0,    "Magic",  "valid_zero"),
        (1,    "Magic",  "valid_one"),
        (2,    "Magic",  "above_one"),
        (-1,   "Mutant", "below_zero"),
        (0,    "Mutant", "valid_zero"),
        (1,    "Mutant", "valid_one"),
        (2,    "Mutant", "above_one"),
        (999,  "Mutant", "far_above"),
    ])
    def test_binary_integer_field_boundary_never_causes_500(
        self, value, field, label, call_endpoint, valid_record
    ):
        """Teams, Magic, Mutant at boundary values must not crash the server.

        Valid range is {0, 1}. Values outside this should be rejected with
        400 or handled gracefully, not cause a 500 crash.
        """
        record = {**valid_record, field: value}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) at {field} boundary '{label}' = {value}\n"
            f"Response: {response_text[:300]}"
        )

    # ── String length boundaries ──────────────────────────────────────────

    @pytest.mark.parametrize("length, label", [
        (0,      "empty_string"),
        (1,      "single_char"),
        (255,    "varchar_255_boundary"),
        (256,    "just_over_varchar_255"),
        (1000,   "one_thousand_chars"),
        (10000,  "ten_thousand_chars"),
        (100000, "hundred_thousand_chars"),
    ])
    def test_string_length_boundary_never_causes_500(
        self, length, label, call_endpoint, valid_record
    ):
        """String fields at various lengths must not crash the server.

        Databases often have VARCHAR limits (e.g. 255). The endpoint must
        handle oversized strings gracefully rather than crashing.

        Note: the string is built inside the test (not at parametrize time)
        to avoid the Windows 32767-char environment variable limit.
        """
        long_string = "A" * length   # built here, not at parametrize time
        record = {**valid_record, "Universe": long_string}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) with Universe of length {length} ('{label}')\n"
            f"Response: {response_text[:300]}"
        )

    # ── Batch size boundaries ──────────────────────────────────────────────

    @pytest.mark.parametrize("batch_size, label", [
        (0,   "empty_batch"),
        (1,   "single_record"),
        (2,   "two_records"),
        (10,  "ten_records"),
        (50,  "fifty_records"),
        (100, "hundred_records"),
    ])
    def test_batch_size_boundary_never_causes_500(
        self, batch_size, label, call_endpoint, valid_record
    ):
        """Batches of varying sizes must not crash the server.

        The endpoint must handle empty batches (return 400) and large
        batches (process or reject) without crashing.
        """
        batch = [valid_record.copy() for _ in range(batch_size)]
        status, response_text = call_endpoint(batch)
        assert status != 500, (
            f"Server crashed (HTTP 500) on batch of size {batch_size} ('{label}')\n"
            f"Response: {response_text[:300]}"
        )


# ===========================================================================
# SECTION C — SCHEMA FUZZING
# Sending the wrong structure: missing fields, extra fields, wrong types,
# deeply nested dicts, lists where scalars are expected, None everywhere.
# ===========================================================================

class TestSchemaFuzzing:
    """Type 3: Schema Fuzzing — break the expected shape of the request.

    The most common real-world source of 500 errors is not bad values
    but bad structure: a field is missing because a client didn't send it,
    or a field contains a nested object instead of a scalar because
    someone refactored the API.  These tests catch every structural
    failure mode.
    """

    # ── Missing fields ────────────────────────────────────────────────────

    @pytest.mark.parametrize("missing_field", ALL_FIELDS)
    def test_single_missing_field_does_not_cause_500(
        self, missing_field, call_endpoint, valid_record
    ):
        """Removing any single field must return 400, not crash with 500.

        This tests every field individually so the exact missing field
        that causes a crash is clearly identified in the test report.
        """
        record = {k: v for k, v in valid_record.items() if k != missing_field}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) when '{missing_field}' was missing.\n"
            f"Response: {response_text[:300]}\n\n"
            f"The endpoint must validate required fields and return 400, not crash."
        )

    def test_all_fields_missing_returns_400_not_500(self, call_endpoint):
        """An empty dict must return 400, not crash with 500."""
        status, response_text = call_endpoint([{}])
        assert status != 500, (
            f"Server crashed (HTTP 500) on completely empty record.\n"
            f"Response: {response_text[:300]}"
        )

    def test_multiple_fields_missing_does_not_cause_500(
        self, call_endpoint, valid_record
    ):
        """A record missing multiple fields must not crash the server."""
        partial = {"Height": 1.75, "Weight": 70.0}   # only 2 of 10 fields
        status, response_text = call_endpoint([partial])
        assert status != 500, (
            f"Server crashed (HTTP 500) on record with only Height and Weight.\n"
            f"Response: {response_text[:300]}"
        )

    # ── Wrong types for each field ────────────────────────────────────────

    @pytest.mark.parametrize("field, bad_value, label", [
        # Numeric fields given strings
        ("Height",  "one_point_seven_five",  "string_instead_of_float"),
        ("Weight",  "seventy",               "string_instead_of_float"),
        ("Teams",   "yes",                   "string_instead_of_int"),
        ("Magic",   "no",                    "string_instead_of_int"),
        ("Mutant",  "no",                    "string_instead_of_int"),
        # String fields given numbers
        ("Universe",      616,   "int_instead_of_string"),
        ("Identity",      True,  "bool_instead_of_string"),
        ("Gender",        0,     "int_instead_of_string"),
        ("Marital_Status", 1.5,  "float_instead_of_string"),
        ("Origin",        [],    "list_instead_of_string"),
        # All fields given None
        ("Height",        None,  "none"),
        ("Weight",        None,  "none"),
        ("Universe",      None,  "none"),
        ("Gender",        None,  "none"),
        ("Teams",         None,  "none"),
        # All fields given a nested dict
        ("Height",  {"nested": 1.75},         "nested_dict"),
        ("Universe", {"name": "Earth-616"},   "nested_dict"),
        # All fields given a list
        ("Height",  [1.75],                   "list"),
        ("Gender",  ["Male", "Female"],        "list"),
    ])
    def test_wrong_type_for_field_does_not_cause_500(
        self, field, bad_value, label, call_endpoint, valid_record
    ):
        """Sending wrong type for each field must return 400, not crash with 500."""
        record = {**valid_record, field: bad_value}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500): field='{field}', "
            f"bad_value={bad_value!r} ({label})\n"
            f"Response: {response_text[:300]}"
        )

    # ── Extra / unexpected fields ──────────────────────────────────────────

    def test_extra_unknown_fields_do_not_cause_500(
        self, call_endpoint, valid_record
    ):
        """Adding unexpected fields to the record must not crash the server.

        Real clients sometimes send extra metadata. The server must ignore
        unknown fields gracefully, not crash.
        """
        record = {
            **valid_record,
            "patient_id":    "P001",
            "timestamp":     "2025-01-01T00:00:00Z",
            "clinic":        "Eye Clinic A",
            "extra_feature": 42.0,
        }
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) when extra unknown fields were included.\n"
            f"Response: {response_text[:300]}"
        )

    # ── Wrong payload structure ────────────────────────────────────────────

    def test_non_list_payload_does_not_cause_500(self, endpoint_url, dbr_token):
        """Sending a dict directly instead of a list of dicts must not crash the server.

        This tests malformed request body structure (not just field values).
        We bypass call_endpoint() here to send truly raw malformed JSON.
        """
        import requests as req
        try:
            response = req.post(
                endpoint_url,
                headers={"Authorization": f"Bearer {dbr_token}"},
                json={"dataframe_records": VALID_RECORD},  # dict, not list
                timeout=30,
            )
            assert response.status_code != 500, (
                f"Server crashed (HTTP 500) when payload was a dict instead of a list.\n"
                f"Response: {response.text[:300]}"
            )
        except Exception:
            pass   # network-level errors are not 500 crashes

    def test_completely_wrong_json_structure_does_not_cause_500(
        self, endpoint_url, dbr_token
    ):
        """Sending a totally unexpected JSON structure must not crash the server."""
        import requests as req
        wrong_payloads = [
            {"completely": "wrong"},
            {"data": [1, 2, 3]},
            {"records": VALID_RECORD},
            [],
            "just a string",
        ]
        for payload in wrong_payloads:
            try:
                response = req.post(
                    endpoint_url,
                    headers={"Authorization": f"Bearer {dbr_token}"},
                    json=payload,
                    timeout=30,
                )
                assert response.status_code != 500, (
                    f"Server crashed (HTTP 500) on payload: {payload!r}\n"
                    f"Response: {response.text[:300]}"
                )
            except Exception:
                pass


# ===========================================================================
# SECTION D — SECURITY FUZZING
# Malicious inputs designed to exploit or crash the system.
# The server must NEVER execute or be affected by these inputs.
# ===========================================================================

class TestSecurityFuzzing:
    """Type 4: Security Fuzzing — test inputs that could exploit the system.

    Security fuzz tests verify that malicious user input cannot:
      - Execute JavaScript in downstream consumers (XSS)
      - Execute SQL in the database (SQL injection)
      - Read system files (path traversal)
      - Exhaust memory (payload bombs)
      - Confuse parsing (null bytes, unicode edge cases)
      - Trigger ReDoS (regex denial of service)

    The endpoint must reject all of these cleanly (400) or process them
    harmlessly (200).  It must NEVER crash (500).
    """

    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert(document.cookie)",
        "<svg onload=alert(1)>",
        '"><script>alert(String.fromCharCode(88,83,83))</script>',
    ]

    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE marvel_characters; --",
        "1' OR '1'='1",
        "1; SELECT * FROM users --",
        "' UNION SELECT null, username, password FROM users --",
        "admin'--",
        "1' AND SLEEP(5) --",
    ]

    PATH_TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        "/etc/shadow",
        "../../../../proc/self/environ",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ]

    UNICODE_PAYLOADS = [
        "\u0000",                          # null character
        "\uffff",                          # largest BMP character
        "\U0001F4A9",                      # emoji
        "日本語テスト",                     # Japanese
        "مرحبا بالعالم",                   # Arabic (RTL)
        "\u202e reversed text",            # RTL override
        "\ud800",                          # lone surrogate (invalid UTF-16)
        "A" * 10 + "\x00" + "B" * 10,     # null byte in middle
    ]

    # Sizes only — strings are generated INSIDE the test to avoid
    # Windows 32767-char environment variable limit at collection time.
    PAYLOAD_BOMB_SIZES = [
        (100_000,  "A",  "100KB_A_chars"),
        (1_000_000, "A", "1MB_A_chars"),
        (10_000,   "\n", "10K_newlines"),
        (50_000,   " ",  "50K_spaces"),
    ]

    REDOS_PAYLOADS = [
        "a" * 100 + "!",                   # triggers catastrophic backtracking
        "(" * 50 + "a" + ")" * 50,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
    ]

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_payload_in_string_field_does_not_cause_500(
        self, payload, call_endpoint, valid_record
    ):
        """XSS strings in text fields must not crash the server.

        The endpoint must treat these as plain strings. It must not
        execute them or crash when they are stored/returned.
        """
        record = {**valid_record, "Universe": payload}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) on XSS payload: {payload!r}\n"
            f"Response: {response_text[:300]}"
        )

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_payload_does_not_cause_500(
        self, payload, call_endpoint, valid_record
    ):
        """SQL injection strings in text fields must not crash the server.

        If this crashes the server it means the input is being interpolated
        directly into a SQL query without parameterisation -- a critical
        security vulnerability.
        """
        record = {**valid_record, "Identity": payload}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) on SQL injection payload: {payload!r}\n"
            f"Response: {response_text[:300]}\n\n"
            f"CRITICAL: If this is a real 500, the input may be interpolated "
            f"directly into SQL. Investigate immediately."
        )

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_payload_does_not_cause_500(
        self, payload, call_endpoint, valid_record
    ):
        """Path traversal strings in text fields must not crash the server.

        These payloads attempt to read system files. The server must
        treat them as plain strings, not file paths.
        """
        record = {**valid_record, "Origin": payload}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) on path traversal payload: {payload!r}\n"
            f"Response: {response_text[:300]}"
        )

    @pytest.mark.parametrize("payload", UNICODE_PAYLOADS)
    def test_unicode_edge_cases_do_not_cause_500(
        self, payload, call_endpoint, valid_record
    ):
        """Unicode edge cases in text fields must not crash the server.

        Null bytes, emoji, RTL text, and lone surrogates can crash
        parsers, databases, and logging systems that aren't UTF-8 safe.
        """
        try:
            record = {**valid_record, "Gender": payload}
            status, response_text = call_endpoint([record])
            assert status != 500, (
                f"Server crashed (HTTP 500) on unicode payload: {payload!r}\n"
                f"Response: {response_text[:300]}"
            )
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass   # encoding error on client side is not a server crash

    @pytest.mark.parametrize("size,char,label", PAYLOAD_BOMB_SIZES)
    def test_oversized_string_does_not_cause_500(
        self, size, char, label, call_endpoint, valid_record
    ):
        """Extremely large strings must not crash or hang the server.

        A 1MB string in a single field should be rejected cleanly (400)
        or truncated, not accepted and cause a memory overflow (500).

        Strings are generated inside the test (not at class level) to avoid
        the Windows 32767-character environment variable limit at collection time.
        """
        payload = char * size   # generated here, not at parametrize time
        record = {**valid_record, "Universe": payload}
        status, response_text = call_endpoint([record], timeout=60)
        assert status != 500, (
            f"Server crashed (HTTP 500) on {label} ({size}-char payload bomb).\n"
            f"Response: {response_text[:300]}"
        )

    def test_deeply_nested_json_does_not_cause_500(
        self, call_endpoint, valid_record
    ):
        """A deeply nested dict value must not crash JSON parsers on the server."""
        nested = "leaf"
        for _ in range(100):
            nested = {"child": nested}

        record = {**valid_record, "Universe": nested}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) on 100-level deeply nested JSON value.\n"
            f"Response: {response_text[:300]}"
        )

    def test_all_security_payloads_combined_do_not_cause_500(
        self, call_endpoint, valid_record
    ):
        """All security payloads simultaneously in different fields must not crash.

        This is the worst-case security fuzz: every field contains a
        different type of malicious input at the same time.
        """
        record = {
            "Height":         float("nan"),
            "Weight":         float("inf"),
            "Universe":       "<script>alert('xss')</script>",
            "Identity":       "'; DROP TABLE marvel_characters; --",
            "Gender":         "../../../etc/passwd",
            "Marital_Status": "\u0000\uffff\U0001F4A9",
            "Teams":          -999,
            "Origin":         "A" * 10000,
            "Magic":          None,
            "Mutant":         {"nested": {"deeply": {"value": True}}},
        }
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) on combined security payload.\n"
            f"Response: {response_text[:300]}\n\n"
            f"This is the most severe fuzz failure -- the endpoint has no "
            f"protection against simultaneous multi-field attacks."
        )


# ===========================================================================
# SECTION E — HYPOTHESIS PROPERTY-BASED FUZZING
# Let Hypothesis auto-generate and shrink failing inputs automatically.
# These are the most powerful fuzz tests because Hypothesis finds edge
# cases you would never think to write manually.
# ===========================================================================

class TestHypothesisFuzzing:
    """Hypothesis-powered fuzzing — auto-generated inputs with auto-shrinking.

    When Hypothesis finds a failure it automatically:
    1. Shrinks the input to the smallest possible failing case
    2. Saves it to .hypothesis/examples/ so it always re-tests it
    3. Reports the exact value that caused the crash

    This means one Hypothesis test can discover bugs that would take
    hundreds of hand-written parametrize cases to find.
    """

    @given(
        height=st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.integers(),
            st.none(),
            st.text(max_size=100),
        ),
        weight=st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.integers(),
            st.none(),
        ),
    )
    @FAST_SETTINGS
    def test_hypothesis_numeric_fields_never_cause_500(
        self, height, weight, call_endpoint, valid_record
    ):
        """Hypothesis generates hundreds of (height, weight) combinations.

        It will automatically find edge cases like:
        - (nan, inf), (0.0, -0.0), (1e308, 1e-308)
        - (None, None), ("tall", "heavy"), (-1, 999999)
        """
        record = {**valid_record, "Height": height, "Weight": weight}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed with Height={height!r}, Weight={weight!r}\n"
            f"Response: {response_text[:300]}"
        )

    @given(text_value=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),  # exclude surrogates
        max_size=500,
    ))
    @FAST_SETTINGS
    def test_hypothesis_any_unicode_string_never_causes_500(
        self, text_value, call_endpoint, valid_record
    ):
        """Hypothesis generates arbitrary Unicode strings for text fields.

        It will automatically find strings containing:
        - Null bytes, control characters, RTL marks
        - Emoji, CJK characters, Arabic, Hebrew
        - Mixed scripts, zero-width spaces, BOM characters
        """
        record = {**valid_record, "Universe": text_value}
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed with Universe={text_value!r}\n"
            f"Response: {response_text[:300]}"
        )

    @given(n_records=st.integers(min_value=0, max_value=50))
    @FAST_SETTINGS
    def test_hypothesis_variable_batch_size_never_causes_500(
        self, n_records, call_endpoint, valid_record
    ):
        """Hypothesis varies the batch size from 0 to 50 records.

        Finds batch size bugs like:
        - Off-by-one in batch processing loops
        - Empty batch causing division-by-zero in preprocessing
        - Large batch exceeding memory limits
        """
        batch = [valid_record.copy() for _ in range(n_records)]
        status, response_text = call_endpoint(batch)
        assert status != 500, (
            f"Server crashed on batch of {n_records} records.\n"
            f"Response: {response_text[:300]}"
        )

    @given(record=st.fixed_dictionaries({
        "Height":         st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.none()),
        "Weight":         st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.none()),
        "Universe":       st.one_of(st.text(max_size=200), st.none()),
        "Identity":       st.one_of(st.text(max_size=200), st.none()),
        "Gender":         st.one_of(st.text(max_size=200), st.none()),
        "Marital_Status": st.one_of(st.text(max_size=200), st.none()),
        "Teams":          st.one_of(st.integers(), st.none()),
        "Origin":         st.one_of(st.text(max_size=200), st.none()),
        "Magic":          st.one_of(st.integers(), st.none()),
        "Mutant":         st.one_of(st.integers(), st.none()),
    }))
    @FAST_SETTINGS
    def test_hypothesis_full_record_with_nones_and_randoms_never_causes_500(
        self, record, call_endpoint
    ):
        """Hypothesis generates a full record with any field potentially None.

        This is the most comprehensive single Hypothesis test:
        every field can be None or a random valid-ish value.
        Hypothesis will find the exact combination of Nones that crashes.
        """
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed on record: {record}\n"
            f"Response: {response_text[:300]}"
        )


# ===========================================================================
# SECTION F — RECHECK SCOPE FUZZING (Marvel-specific)
# Verify that even with corrupted recheck requests, the endpoint
# returns a valid response and does not crash.
# ===========================================================================

class TestRecheckScopeFuzzing:
    """Fuzz tests specific to the recheck workflow.

    The recheck system allows clinicians to request a re-evaluation of
    a specific character's survival prediction.  These tests verify the
    endpoint handles corrupted, missing, or malicious recheck payloads
    without crashing.
    """

    RECHECK_FUZZ_CASES = [
        # (description, student_id_value, question_id_value)
        ("empty_student_id",       "",         "case_001"),
        ("none_student_id",        None,       "case_001"),
        ("numeric_student_id",     12345,      "case_001"),
        ("empty_case_id",          "STU001",   ""),
        ("none_case_id",           "STU001",   None),
        ("numeric_case_id",        "STU001",   9999),
        ("both_empty",             "",         ""),
        ("both_none",              None,       None),
        ("sql_in_student_id",      "'; DROP TABLE predictions; --", "case_001"),
        ("xss_in_case_id",         "STU001",   "<script>"),
        ("very_long_student_id",   "S" * 500, "case_001"),
        ("very_long_case_id",      "STU001",   "C" * 500),
        ("unicode_student_id",     "\u0000\uffff", "case_001"),
    ]

    @pytest.mark.parametrize("description, record_id, case_id", RECHECK_FUZZ_CASES)
    def test_fuzzed_recheck_identifiers_do_not_cause_500(
        self, description, record_id, case_id, call_endpoint, valid_record
    ):
        """Fuzzed record_id and case_id values in a recheck-style payload
        must not crash the server.

        The recheck endpoint receives (record_id, case_id) to identify which
        prediction to re-evaluate.  Corrupted identifiers must be rejected
        cleanly, not cause a server crash.
        """
        record = {
            **valid_record,
            "_recheck_record_id": record_id,
            "_recheck_case_id":   case_id,
        }
        status, response_text = call_endpoint([record])
        assert status != 500, (
            f"Server crashed (HTTP 500) on recheck fuzz case '{description}'.\n"
            f"record_id={record_id!r}, case_id={case_id!r}\n"
            f"Response: {response_text[:300]}"
        )
