"""Property-based tests using Hypothesis for core invariants.

Invariants verified:
1. All score/confidence outputs are always in [0, 1]
2. Serialization round-trips preserved (DetectionResult fields)
3. No crashes on any random valid input
4. is_hallucination is always a bool
5. Weighted score respects threshold boundary
6. Batch results always same length as inputs
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from groundtruth import (
    ConsistencyDetector,
    DetectionResult,
    EntailmentDetector,
    GroundTruthDetector,
    OverlapDetector,
)
from groundtruth.detectors.consistency import _jaccard
from groundtruth.detectors.overlap import _ngram_recall

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),  # exclude surrogates
    min_size=0,
    max_size=200,
)

short_text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=0,
    max_size=50,
)

positive_float = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)

threshold_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 1: Scores always in [0, 1]
# ---------------------------------------------------------------------------


@given(claim=text_strategy, context=text_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_overlap_score_always_in_unit_interval(claim: str, context: str) -> None:
    """OverlapDetector score is always in [0, 1] for any valid text input."""
    det = OverlapDetector()
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


@given(claim=text_strategy, context=text_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_entailment_score_always_in_unit_interval(claim: str, context: str) -> None:
    """EntailmentDetector score is always in [0, 1] for any valid text input."""
    det = EntailmentDetector()
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


@given(claim=text_strategy, context=text_strategy)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_consistency_score_always_in_unit_interval(claim: str, context: str) -> None:
    """ConsistencyDetector score is always in [0, 1] for any valid text input."""
    det = ConsistencyDetector()
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


@given(claim=text_strategy, context=text_strategy)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_aggregator_score_always_in_unit_interval(claim: str, context: str) -> None:
    """GroundTruthDetector aggregated score is always in [0, 1]."""
    det = GroundTruthDetector()
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Property 2: is_hallucination is always bool and consistent with threshold
# ---------------------------------------------------------------------------


@given(claim=text_strategy, context=text_strategy, threshold=threshold_strategy)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_is_hallucination_consistent_with_threshold(
    claim: str, context: str, threshold: float
) -> None:
    """is_hallucination == (score > threshold) always holds."""
    det = GroundTruthDetector(threshold=threshold)
    result = det.detect(claim=claim, context=context)
    assert isinstance(result.is_hallucination, bool)
    assert result.is_hallucination == (result.score > threshold)


# ---------------------------------------------------------------------------
# Property 3: No crashes on any random valid string input
# ---------------------------------------------------------------------------


@given(claim=text_strategy, context=text_strategy)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_overlap_never_crashes(claim: str, context: str) -> None:
    """OverlapDetector never raises on any valid string input."""
    det = OverlapDetector()
    det.detect(claim=claim, context=context)  # must not raise


@given(claim=text_strategy, context=text_strategy)
@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
def test_entailment_never_crashes(claim: str, context: str) -> None:
    """EntailmentDetector never raises on any valid string input."""
    det = EntailmentDetector()
    det.detect(claim=claim, context=context)  # must not raise


@given(
    claim=text_strategy,
    context=text_strategy,
    candidates=st.lists(short_text_strategy, min_size=0, max_size=10),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_consistency_never_crashes(claim: str, context: str, candidates: list[str]) -> None:
    """ConsistencyDetector never raises on any valid string input + candidate list."""
    det = ConsistencyDetector()
    det.detect(claim=claim, context=context, candidates=candidates if candidates else None)


# ---------------------------------------------------------------------------
# Property 4: DetectionResult serialisation round-trip
# ---------------------------------------------------------------------------


@given(
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    is_hallucination=st.booleans(),
    method=st.text(min_size=1, max_size=50),
)
def test_detection_result_round_trip(
    score: float, confidence: float, is_hallucination: bool, method: str
) -> None:
    """DetectionResult fields survive an asdict() round-trip unchanged."""
    r = DetectionResult(
        score=score,
        is_hallucination=is_hallucination,
        confidence=confidence,
        method=method,
    )
    d = asdict(r)
    assert d["score"] == r.score
    assert d["confidence"] == r.confidence
    assert d["is_hallucination"] == r.is_hallucination
    assert d["method"] == r.method
    assert d["details"] == {}


# ---------------------------------------------------------------------------
# Property 5: Batch results always same length as inputs
# ---------------------------------------------------------------------------


@given(
    pairs=st.lists(
        st.tuples(short_text_strategy, short_text_strategy),
        min_size=0,
        max_size=10,
    )
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_batch_length_preserved(pairs: list[tuple[str, str]]) -> None:
    """detect_batch always returns same number of results as input pairs."""
    det = GroundTruthDetector(detectors=[OverlapDetector()])
    claims = [p[0] for p in pairs]
    contexts = [p[1] for p in pairs]
    results = det.detect_batch(claims, contexts)
    assert len(results) == len(pairs)


# ---------------------------------------------------------------------------
# Property 6: _jaccard symmetry
# ---------------------------------------------------------------------------


@given(a=short_text_strategy, b=short_text_strategy)
@settings(max_examples=200)
def test_jaccard_symmetric(a: str, b: str) -> None:
    """Jaccard similarity is symmetric: J(a,b) == J(b,a)."""
    assert _jaccard(a, b) == pytest.approx(_jaccard(b, a))


@given(a=short_text_strategy)
@settings(max_examples=100)
def test_jaccard_self_similarity_is_one(a: str) -> None:
    """Jaccard similarity of any text with itself is 1.0."""
    result = _jaccard(a, a)
    assert result == pytest.approx(1.0)


@given(a=short_text_strategy, b=short_text_strategy)
@settings(max_examples=200)
def test_jaccard_in_unit_interval(a: str, b: str) -> None:
    """Jaccard similarity is always in [0, 1]."""
    result = _jaccard(a, b)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Property 7: _ngram_recall in [0, 1]
# ---------------------------------------------------------------------------


@given(
    claim=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=20),
    context=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=20),
    n=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=200)
def test_ngram_recall_in_unit_interval(claim: list[str], context: list[str], n: int) -> None:
    """_ngram_recall always returns a value in [0, 1]."""
    result = _ngram_recall(claim, context, n)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Property 8: OverlapDetector threshold monotonicity
# ---------------------------------------------------------------------------


@given(claim=short_text_strategy, context=short_text_strategy)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_overlap_threshold_monotonicity(claim: str, context: str) -> None:
    """Higher threshold → more likely to flag hallucination (monotone property)."""
    det_strict = OverlapDetector(threshold=0.9)
    det_lax = OverlapDetector(threshold=0.1)

    r_strict = det_strict.detect(claim=claim, context=context)
    r_lax = det_lax.detect(claim=claim, context=context)

    # If strict flags hallucination, lax must also agree OR have the same score
    # (scores are identical; only threshold differs)
    assert r_strict.score == pytest.approx(r_lax.score)
    # Strict threshold should flag at least as often as lax threshold
    if r_lax.is_hallucination:
        assert r_strict.is_hallucination
