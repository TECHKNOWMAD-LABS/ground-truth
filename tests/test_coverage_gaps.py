"""Tests targeting specific coverage gaps: aggregator zero-weights, overlap single-token,
consistency jaccard edge cases."""

from __future__ import annotations

import pytest

from groundtruth import (
    EntailmentDetector,
    GroundTruthDetector,
    OverlapDetector,
)
from groundtruth.detectors.consistency import _jaccard  # noqa: PLC2701
from groundtruth.detectors.overlap import _ngram_recall, _tokenize  # noqa: PLC2701
from tests.conftest import ClampingNLI, FixedNLI

# ---------------------------------------------------------------------------
# aggregator.py line 53 — zero weights sum
# ---------------------------------------------------------------------------


def test_aggregator_zero_weights_raises() -> None:
    """All-zero weights must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        GroundTruthDetector(
            detectors=[OverlapDetector()],
            weights=[0.0],
        )


# ---------------------------------------------------------------------------
# aggregator.py line 121 — add_detector with zero/negative weight
# ---------------------------------------------------------------------------


def test_aggregator_add_detector_zero_weight_raises() -> None:
    """add_detector with weight <= 0 must raise ValueError."""
    agg = GroundTruthDetector(detectors=[OverlapDetector()])
    with pytest.raises(ValueError, match="positive"):
        agg.add_detector(EntailmentDetector(), weight=0.0)


def test_aggregator_add_detector_negative_weight_raises() -> None:
    """add_detector with negative weight must raise ValueError."""
    agg = GroundTruthDetector(detectors=[OverlapDetector()])
    with pytest.raises(ValueError, match="positive"):
        agg.add_detector(EntailmentDetector(), weight=-1.0)


# ---------------------------------------------------------------------------
# overlap.py line 22 — _ngram_recall with empty claim n-grams
# ---------------------------------------------------------------------------


def test_ngram_recall_empty_claim_returns_zero() -> None:
    """_ngram_recall with empty claim tokens returns 0.0 (not division error)."""
    # Single token cannot form a 2-gram → claim_ng is empty
    result = _ngram_recall(["word"], ["word", "other"], n=2)
    assert result == 0.0


def test_ngram_recall_full_match() -> None:
    """All claim n-grams present in context → recall 1.0."""
    claim = ["the", "cat", "sat"]
    context = ["the", "cat", "sat", "on", "mat"]
    result = _ngram_recall(claim, context, n=2)
    assert result == pytest.approx(1.0)


def test_ngram_recall_no_match() -> None:
    """No claim n-grams present in context → recall 0.0."""
    result = _ngram_recall(["dog", "runs"], ["cat", "sleeps"], n=2)
    assert result == pytest.approx(0.0)


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    """_tokenize should lower-case and extract word tokens."""
    tokens = _tokenize("Hello, World! This is a TEST.")
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


# ---------------------------------------------------------------------------
# consistency.py lines 15–17 — _jaccard edge cases
# ---------------------------------------------------------------------------


def test_jaccard_both_empty_returns_one() -> None:
    """Both empty texts → Jaccard is 1.0 (both have empty token sets)."""
    result = _jaccard("", "")
    assert result == pytest.approx(1.0)


def test_jaccard_one_empty_returns_zero() -> None:
    """One empty text → Jaccard is 0.0."""
    assert _jaccard("hello world", "") == pytest.approx(0.0)
    assert _jaccard("", "hello world") == pytest.approx(0.0)


def test_jaccard_identical_texts() -> None:
    """Identical non-empty texts → Jaccard is 1.0."""
    text = "the quick brown fox"
    assert _jaccard(text, text) == pytest.approx(1.0)


def test_jaccard_disjoint_texts() -> None:
    """Completely different texts → Jaccard is 0.0."""
    assert _jaccard("apple banana", "car truck") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EntailmentDetector clamping (NLI out-of-range)
# ---------------------------------------------------------------------------


def test_entailment_nli_score_clamped_above_one() -> None:
    """NLI provider returning >1.0 must be clamped to 1.0."""
    det = EntailmentDetector(threshold=0.5, nli_provider=ClampingNLI())
    result = det.detect(claim="anything", context="anything")
    assert result.details["entailment_score"] == pytest.approx(1.0)
    assert not result.is_hallucination


# ---------------------------------------------------------------------------
# OverlapDetector — n-gram size larger than claim
# ---------------------------------------------------------------------------


def test_overlap_ngram_larger_than_claim_skipped() -> None:
    """When claim has fewer tokens than n, that n-gram is skipped gracefully."""
    det = OverlapDetector(n_values=(1, 5), threshold=0.5)
    result = det.detect(claim="hi", context="hi there")
    # Only recall_1gram should be in details (5-gram skipped for single token)
    assert "recall_1gram" in result.details
    assert "recall_5gram" not in result.details
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# GroundTruthDetector threshold boundary
# ---------------------------------------------------------------------------


def test_aggregator_score_exactly_at_threshold_not_hallucination() -> None:
    """Score exactly equal to threshold → not hallucination (> threshold required)."""
    # Use a fixed NLI that gives entailment=0.5 → score=0.5, threshold=0.5
    nli = FixedNLI(0.5)
    det = GroundTruthDetector(
        detectors=[EntailmentDetector(threshold=0.5, nli_provider=nli)],
        threshold=0.5,
    )
    result = det.detect(claim="x", context="x")
    # weighted score = 0.5, threshold = 0.5 → score > 0.5 is False
    assert not result.is_hallucination


def test_aggregator_detect_batch_empty_lists() -> None:
    """detect_batch with empty lists returns empty list."""
    agg = GroundTruthDetector()
    results = agg.detect_batch([], [])
    assert results == []
