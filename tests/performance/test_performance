"""Performance / Load Tests for the Marvel Characters serving endpoint.

WHAT IS PERFORMANCE TESTING?
----------------------------
Performance tests check the system handles real-world traffic volumes without
slowing down, crashing, or producing errors.

KEY METRICS
-----------
    Throughput    - requests handled per second
    p50 latency   - median response time
    p95 latency   - 95th percentile response time
    p99 latency   - 99th percentile response time
    Error rate    - % of requests that fail

THRESHOLDS ARE TUNED FOR DATABRICKS SERVERLESS MODEL SERVING
------------------------------------------------------------
Serverless endpoints have network overhead and (after idle) cold-start time,
so a single call realistically takes ~1 second, not the ~200ms you would see
on a warm dedicated endpoint. The thresholds below reflect that reality.

The point of these tests is NOT to hit some universal number -- it is to catch
when YOUR endpoint becomes meaningfully slower than ITS OWN normal. If your
endpoint is consistently faster (e.g. dedicated compute), tighten the numbers.

Every test also does a WARM-UP call first and discards it, so cold-start time
does not pollute the measurement.

The slow / concurrent tests are marked @pytest.mark.slow:
    pytest tests/performance/ -v -m slow
    pytest tests/performance/ -v -m "not slow"
"""

import concurrent.futures
import time

import pytest


# ---------------------------------------------------------------------------
# Thresholds -- tuned for Databricks serverless. Adjust to YOUR endpoint's
# measured baseline. A good rule: set the limit ~2x your observed normal so
# the test still catches a real regression without flapping on noise.
# ---------------------------------------------------------------------------
P50_MS_MAX        = 1500    # median response time ceiling
P95_MS_MAX        = 2500    # 95th percentile ceiling
P99_MS_MAX        = 4000    # 99th percentile ceiling
MIN_THROUGHPUT    = 5       # requests / second floor (serverless-realistic)
MAX_ERROR_RATE    = 0.01    # 1 %

# Number of warm-up calls discarded before each measurement, to remove
# cold-start skew (the endpoint scaling up from zero).
WARMUP_CALLS = 2


def _percentile(sorted_values, pct):
    """Return the pct-th percentile from an already-sorted list."""
    if not sorted_values:
        return 0.0
    idx = min(int(pct * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


def _warm_up(call_endpoint, record, n=WARMUP_CALLS):
    """Fire a few throwaway calls so the endpoint is warm before we measure.

    The very first call to an idle serverless endpoint pays the cold-start
    cost (scaling up from zero). Measuring that would unfairly inflate every
    latency number, so we discard these calls.
    """
    for _ in range(n):
        try:
            call_endpoint([record])
        except Exception:
            pass


class TestLatency:
    """Single-threaded latency percentiles (after warm-up)."""

    def test_latency_percentiles(self, call_endpoint, valid_record):
        """Measure p50/p95/p99 over 20 sequential requests, after warm-up.

        p95 is the headline number: 95% of users wait at most this long.
        """
        _warm_up(call_endpoint, valid_record)

        latencies = []
        for _ in range(20):
            status, _, elapsed_ms = call_endpoint([valid_record])
            assert status == 200, f"Request failed mid-benchmark: HTTP {status}"
            latencies.append(elapsed_ms)

        latencies.sort()
        p50 = _percentile(latencies, 0.50)
        p95 = _percentile(latencies, 0.95)
        p99 = _percentile(latencies, 0.99)

        print(f"\n  p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")

        # p95 is the SLA headline. p50/p99 are printed for visibility but we
        # assert on p95 to avoid flapping on a single slow outlier (p99).
        assert p95 < P95_MS_MAX, (
            f"p95 latency {p95:.0f}ms exceeds {P95_MS_MAX}ms.\n"
            f"  Observed: p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms\n"
            f"  If this is your endpoint's genuine normal, raise P95_MS_MAX. "
            f"If it suddenly regressed, investigate the model / compute."
        )

    def test_median_latency_reasonable(self, call_endpoint, valid_record):
        """The MEDIAN (p50) request should be comfortably under its ceiling.

        p50 is more stable than p95 -- if even the median is slow, the whole
        endpoint is slow, not just a few outliers.
        """
        _warm_up(call_endpoint, valid_record)

        latencies = []
        for _ in range(15):
            status, _, elapsed_ms = call_endpoint([valid_record])
            assert status == 200
            latencies.append(elapsed_ms)
        latencies.sort()
        p50 = _percentile(latencies, 0.50)
        print(f"\n  p50={p50:.0f}ms")
        assert p50 < P50_MS_MAX, (
            f"Median latency {p50:.0f}ms exceeds {P50_MS_MAX}ms. "
            f"The endpoint is slow even at the median."
        )


class TestThroughput:
    """Concurrent throughput and error-rate under load (after warm-up)."""

    @pytest.mark.slow
    def test_minimum_throughput(self, call_endpoint, valid_record):
        """Fire 30 requests across 10 workers; throughput must clear the floor.

        Warm-up first so the cold-start call is not counted against throughput.
        """
        _warm_up(call_endpoint, valid_record)

        n = 30
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(call_endpoint, [valid_record]) for _ in range(n)]
            results = [f.result() for f in futures]
        duration = time.perf_counter() - start

        throughput = n / duration
        errors = sum(1 for status, _, _ in results if status != 200)

        print(f"\n  throughput={throughput:.1f} req/s  errors={errors}/{n}  duration={duration:.1f}s")

        assert throughput >= MIN_THROUGHPUT, (
            f"Throughput {throughput:.1f} req/s is below the {MIN_THROUGHPUT} req/s floor.\n"
            f"  If this is your serverless endpoint's genuine capacity, lower "
            f"MIN_THROUGHPUT. If it regressed, check compute scaling / autoscaling config."
        )

    @pytest.mark.slow
    def test_error_rate_under_concurrent_load(self, call_endpoint, valid_record):
        """50 concurrent requests must keep the error rate under the SLA."""
        _warm_up(call_endpoint, valid_record)

        n = 50
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as ex:
            futures = [ex.submit(call_endpoint, [valid_record]) for _ in range(n)]
            results = [f.result() for f in futures]

        errors = sum(1 for status, _, _ in results if status != 200)
        error_rate = errors / n
        print(f"\n  error_rate={error_rate:.1%} ({errors}/{n})")

        assert error_rate <= MAX_ERROR_RATE, (
            f"Error rate {error_rate:.1%} exceeds the {MAX_ERROR_RATE:.0%} SLA "
            f"under {n} concurrent requests."
        )

    @pytest.mark.slow
    def test_no_server_crashes_under_load(self, call_endpoint, valid_record):
        """Under 50 concurrent requests, zero responses may be HTTP 500."""
        _warm_up(call_endpoint, valid_record)

        n = 50
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as ex:
            futures = [ex.submit(call_endpoint, [valid_record]) for _ in range(n)]
            results = [f.result() for f in futures]
        crashes = sum(1 for status, _, _ in results if status == 500)
        assert crashes == 0, f"{crashes}/{n} requests crashed the server (HTTP 500)."


class TestBatchEfficiency:
    """Batching should be more efficient than one-by-one calls."""

    @pytest.mark.slow
    def test_batch_latency_reasonable(self, call_endpoint, record_bank, valid_record):
        """A single batch of 8 records should cost far less than 8 separate
        calls -- proving the endpoint batches efficiently rather than looping."""
        _warm_up(call_endpoint, valid_record)

        records = list(record_bank.values())
        status, _, batch_ms = call_endpoint(records)
        assert status == 200

        # one single-record call for reference (also warm)
        _, _, single_ms = call_endpoint([records[0]])

        print(f"\n  batch_of_{len(records)}={batch_ms:.0f}ms  single={single_ms:.0f}ms")
        assert batch_ms < single_ms * len(records), (
            f"Batch of {len(records)} ({batch_ms:.0f}ms) is no faster than "
            f"{len(records)} separate calls ({single_ms:.0f}ms each) -- "
            f"batching is not effective."
        )
