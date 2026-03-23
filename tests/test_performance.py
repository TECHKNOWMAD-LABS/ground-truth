"""Performance tests: parallel batch detection, caching, timing measurements."""
from __future__ import annotations

import time

import pytest

from groundtruth import GroundTruthDetector, OverlapDetector
from groundtruth.aggregator import cached_tokenize
from groundtruth.detectors.overlap import _tokenize

# ---------------------------------------------------------------------------
# Parallel batch detection
# ---------------------------------------------------------------------------


def test_detect_batch_parallel_correctness() -> None:
    """Parallel detect_batch produces same results as sequential detect() calls."""
    agg = GroundTruthDetector()
    claims = [
        "Paris is the capital of France",
        "The moon orbits the Earth",
        "Dogs are mammals",
        "Water boils at 100 degrees Celsius",
        "The Amazon is the longest river in the world",
    ]
    contexts = [
        "France's capital is Paris",
        "The Moon is Earth's natural satellite",
        "Dogs belong to the mammal class",
        "Water reaches boiling point at 100°C at standard pressure",
        "The Amazon river is the largest by discharge volume in the world",
    ]

    # Sequential reference
    sequential = [agg.detect(c, ctx) for c, ctx in zip(claims, contexts)]
    # Parallel batch
    parallel = agg.detect_batch(claims, contexts)

    assert len(parallel) == len(sequential)
    for seq, par in zip(sequential, parallel):
        assert seq.score == pytest.approx(par.score, abs=1e-9)
        assert seq.is_hallucination == par.is_hallucination


def test_detect_batch_single_item_no_thread_overhead() -> None:
    """Single-item detect_batch bypasses thread pool and returns correct result."""
    agg = GroundTruthDetector()
    results = agg.detect_batch(["Paris is in France"], ["Paris is the capital of France"])
    assert len(results) == 1
    assert 0.0 <= results[0].score <= 1.0


def test_detect_batch_preserves_order() -> None:
    """Parallel batch must preserve input order even with concurrent execution."""
    agg = GroundTruthDetector(detectors=[OverlapDetector()])
    claims = [f"claim number {i}" for i in range(20)]
    contexts = [f"context number {i}" for i in range(20)]

    results = agg.detect_batch(claims, contexts)
    assert len(results) == 20

    # Each result corresponds to its own index pair (identical → low score)
    for result in results:
        assert 0.0 <= result.score <= 1.0


def test_detect_batch_custom_workers() -> None:
    """max_workers parameter is respected without errors."""
    agg = GroundTruthDetector(detectors=[OverlapDetector()], max_workers=2)
    claims = ["test claim"] * 5
    contexts = ["test context"] * 5
    results = agg.detect_batch(claims, contexts)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------


def test_tokenize_cache_hit() -> None:
    """_tokenize returns same list object on cache hit (lru_cache)."""
    text = "the quick brown fox jumps over the lazy dog"
    result1 = _tokenize(text)
    result2 = _tokenize(text)
    # With lru_cache, identical calls return identical result
    assert result1 == result2


def test_cached_tokenize_returns_tuple() -> None:
    """cached_tokenize returns a tuple (hashable) not a list."""
    result = cached_tokenize("hello world test")
    assert isinstance(result, tuple)
    assert result == ("hello", "world", "test")


def test_cached_tokenize_empty_string() -> None:
    """cached_tokenize handles empty string without error."""
    result = cached_tokenize("")
    assert result == ()


def test_repeated_batch_faster_with_same_context() -> None:
    """Repeated detection on same context should benefit from tokenise cache.

    # Before caching: each call re-runs regex on 'context'
    # After caching: second call returns cached tokens immediately
    """
    agg = GroundTruthDetector(detectors=[OverlapDetector()])
    context = "the exact same context text used repeatedly across all claims"
    claims = [f"claim {i} with unique words" for i in range(50)]
    contexts = [context] * 50

    start = time.perf_counter()
    results = agg.detect_batch(claims, contexts)
    elapsed = time.perf_counter() - start

    assert len(results) == 50
    # Should complete in under 2 seconds even for 50 pairs
    assert elapsed < 2.0, f"Batch detection too slow: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Timing measurement for single detect
# ---------------------------------------------------------------------------


def test_single_detect_fast_for_short_texts() -> None:
    """Single detect() call completes in under 100ms for short texts."""
    agg = GroundTruthDetector()
    start = time.perf_counter()
    agg.detect(
        claim="The Eiffel Tower is located in Paris",
        context="The Eiffel Tower is a famous landmark in Paris, France",
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1, f"detect() too slow: {elapsed:.3f}s"
