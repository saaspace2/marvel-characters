"""Chaos Tests for the Marvel Characters serving endpoint.

WHAT IS CHAOS TESTING?
----------------------
Chaos testing deliberately introduces failures into a live system to verify
it recovers gracefully. If the system survives intentional chaos, it can
survive real-world failures.

    Analogy: Netflix's Chaos Monkey randomly kills production servers during
    business hours. Their system is built to survive it automatically. If it
    cannot survive intentional chaos, it cannot survive real outages.

NOTE
----
Chaos testing is for MATURE systems. Run it only once your smoke, integration,
and serving tests pass reliably. These tests do controlled, safe chaos against
the endpoint -- they do not require infrastructure-level fault injection.

WHAT WE VERIFY
--------------
1. Recovery after a forced client-side timeout.
2. No crashes under a sudden traffic spike.
3. Graceful handling of a slow / aborted request.
4. Stability when valid and malformed requests are interleaved.
5. The endpoint recovers to healthy after the chaos ends.

RUN
---
    pytest tests/advanced/test_chaos.py -v -m slow
"""

import concurrent.futures
import time

import pytest


class TestRecoveryAfterTimeout:
    """The endpoint must keep working after a client aborts a request early."""

    def test_recovers_after_client_timeout(self, call_endpoint, valid_record):
        """Force an ultra-short timeout (aborts client-side), then confirm the
        very next normal request succeeds. Proves one bad client does not
        poison the endpoint for the next caller."""
        # Force a timeout with an absurdly small budget.
        status, _, _ = call_endpoint([valid_record], timeout=0.001)
        # status will likely be 408/503 -- that's fine, we just forced an abort.

        # The endpoint must still serve the next request normally.
        status2, _, _ = call_endpoint([valid_record], timeout=30)
        assert status2 == 200, (
            f"Endpoint did not recover after a client timeout (got HTTP {status2})."
        )


class TestTrafficSpike:
    """A sudden burst of traffic must not crash the endpoint."""

    @pytest.mark.slow
    def test_traffic_spike_no_crashes(self, call_endpoint, valid_record):
        """Fire 60 requests at once (a spike) and assert zero HTTP 500s.

        Real traffic is bursty -- an exam ending, a clinic opening. The
        endpoint must absorb the spike by queueing or scaling, never by
        crashing.
        """
        n = 60
        with concurrent.futures.ThreadPoolExecutor(max_workers=60) as ex:
            futures = [ex.submit(call_endpoint, [valid_record]) for _ in range(n)]
            results = [f.result() for f in futures]
        crashes = sum(1 for status, _, _ in results if status == 500)
        assert crashes == 0, f"{crashes}/{n} requests crashed during the traffic spike."


class TestInterleavedChaos:
    """Mixing good and bad requests must not destabilise the endpoint."""

    @pytest.mark.slow
    def test_valid_requests_survive_among_malformed(self, call_endpoint, valid_record):
        """Interleave valid records with malformed ones under concurrency.

        Every VALID request must still succeed even while malformed requests
        are being rejected alongside it. Proves bad inputs do not knock out
        good ones sharing the same workers.
        """
        good = [valid_record]
        bad  = [{"Height": "not_a_number"}]   # malformed; should be rejected, not crash

        def fire(payload):
            return call_endpoint(payload)

        payloads = []
        for _ in range(20):
            payloads.append(good)
            payloads.append(bad)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            results = list(ex.map(fire, payloads))

        # The good requests are at even indices.
        good_statuses = [results[i][0] for i in range(0, len(results), 2)]
        good_failures = [s for s in good_statuses if s != 200]
        # No request of any kind may return 500.
        crashes = sum(1 for status, _, _ in results if status == 500)

        assert crashes == 0, f"{crashes} requests crashed during interleaved chaos."
        assert not good_failures, (
            f"{len(good_failures)} valid requests failed while malformed requests "
            f"were interleaved: statuses {good_failures}"
        )


class TestRecoveryToHealthy:
    """After the chaos ends, the endpoint must return to a healthy baseline."""

    @pytest.mark.slow
    def test_endpoint_healthy_after_chaos(self, call_endpoint, valid_record):
        """Run a burst of load, then confirm a cooldown request is fast and
        successful -- proving the endpoint settles back to normal."""
        # Burst
        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as ex:
            [f.result() for f in [ex.submit(call_endpoint, [valid_record]) for _ in range(40)]]

        # Cooldown
        time.sleep(2)
        status, _, elapsed_ms = call_endpoint([valid_record])
        assert status == 200, "Endpoint did not return to healthy after chaos."
        print(f"\n  post-chaos latency={elapsed_ms:.0f}ms")
