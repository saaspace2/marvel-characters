"""Canary Tests for the Marvel Characters serving endpoint.

WHAT IS CANARY TESTING?
-----------------------
Canary testing releases a new model to a SMALL percentage of traffic first
(e.g. 5%), monitors for problems, and gradually increases the share to 100%
only if everything stays healthy.

    Analogy: coal miners carried canary birds underground. Canaries react to
    toxic gas faster than humans; if the canary died, miners evacuated. In
    software, a small group of users are the canary -- if errors spike, you
    roll back before everyone is affected.

WHAT WE VERIFY
--------------
1. The canary endpoint is alive and never crashes.
2. The canary error rate is not higher than production.
3. The canary latency is acceptable.
4. The canary's prediction distribution is similar to production
   (a wildly different distribution suggests the new model is broken).

CONFIG
------
Set ENDPOINT_URL (current/stable) and ENDPOINT_URL_CANARY (new model).
If ENDPOINT_URL_CANARY is unset, the suite is skipped.

RUN
---
    pytest tests/advanced/test_canary.py -v
"""

import pytest

from conftest import extract_prediction, normalise_prediction


ERROR_RATE_TOLERANCE = 1.10   # canary may have at most 10% more errors than prod
LATENCY_TOLERANCE    = 1.20   # canary may be at most 20% slower than prod
DISTRIBUTION_TOLERANCE = 0.20 # canary alive-rate may differ by at most 0.20


def _alive_rate(call, url, records, named=True):
    """Return the fraction of records predicted 'alive' (1) by an endpoint."""
    alive = 0
    total = 0
    for record in records:
        if named:
            _, body, _ = call(url, [record])
        else:
            _, body, _ = call([record])
        pred = normalise_prediction(extract_prediction(body))
        if pred == 1:
            alive += 1
        total += 1
    return alive / total if total else 0.0


class TestCanaryLiveness:
    """The canary model must be alive and crash-free."""

    def test_canary_endpoint_alive(self, canary_url, call_named_endpoint, valid_record):
        status, _, _ = call_named_endpoint(canary_url, [valid_record])
        assert status == 200, f"Canary endpoint returned HTTP {status}"

    def test_canary_never_crashes_on_bank(self, canary_url, call_named_endpoint, record_bank):
        crashes = [cid for cid, rec in record_bank.items()
                   if call_named_endpoint(canary_url, [rec])[0] == 500]
        assert not crashes, f"Canary crashed on: {crashes}"


class TestCanaryHealth:
    """The canary must not be worse than production on the key health signals."""

    def test_canary_error_rate_not_higher(
        self, call_endpoint, canary_url, call_named_endpoint, record_bank
    ):
        """Canary error rate must be within tolerance of production error rate."""
        records = list(record_bank.values())
        prod_errors   = sum(1 for r in records if call_endpoint([r])[0] != 200)
        canary_errors = sum(1 for r in records if call_named_endpoint(canary_url, [r])[0] != 200)

        prod_rate   = prod_errors / len(records)
        canary_rate = canary_errors / len(records)
        print(f"\n  prod_err={prod_rate:.1%}  canary_err={canary_rate:.1%}")

        # allow a small absolute floor so 0% prod doesn't make any canary error fail
        assert canary_rate <= max(prod_rate * ERROR_RATE_TOLERANCE, 0.05), (
            f"Canary error rate {canary_rate:.1%} exceeds production "
            f"{prod_rate:.1%} beyond tolerance -- roll back the canary."
        )

    def test_canary_latency_acceptable(
        self, call_endpoint, canary_url, call_named_endpoint, valid_record
    ):
        """Canary latency must be within 20% of production latency."""
        prod_times, canary_times = [], []
        for _ in range(5):
            _, _, p = call_endpoint([valid_record]);                    prod_times.append(p)
            _, _, c = call_named_endpoint(canary_url, [valid_record]);   canary_times.append(c)
        prod_avg   = sum(prod_times) / len(prod_times)
        canary_avg = sum(canary_times) / len(canary_times)
        print(f"\n  prod_avg={prod_avg:.0f}ms  canary_avg={canary_avg:.0f}ms")
        assert canary_avg <= prod_avg * LATENCY_TOLERANCE, (
            f"Canary too slow: {canary_avg:.0f}ms vs prod {prod_avg:.0f}ms."
        )

    def test_canary_distribution_similar(
        self, call_endpoint, canary_url, call_named_endpoint, record_bank
    ):
        """The canary's alive/dead split must be close to production's.

        A canary predicting almost everything 'alive' when production is
        balanced is a strong signal the new model is broken, even if no
        single prediction looks invalid.
        """
        records = list(record_bank.values())
        prod_alive   = _alive_rate(call_endpoint, None, records, named=False)
        canary_alive = _alive_rate(call_named_endpoint, canary_url, records, named=True)
        print(f"\n  prod_alive_rate={prod_alive:.2f}  canary_alive_rate={canary_alive:.2f}")
        assert abs(canary_alive - prod_alive) <= DISTRIBUTION_TOLERANCE, (
            f"Canary distribution differs too much: prod alive-rate "
            f"{prod_alive:.2f} vs canary {canary_alive:.2f}."
        )
