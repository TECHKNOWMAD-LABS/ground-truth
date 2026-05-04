"""Tests for DetectionResult and AggregatedResult models — covers validation branches."""

from __future__ import annotations

import pytest

from groundtruth import AggregatedResult, DetectionResult

# ---------------------------------------------------------------------------
# DetectionResult validation
# ---------------------------------------------------------------------------


def test_detection_result_valid_boundary_values() -> None:
    """score and confidence at exact 0.0 and 1.0 are valid."""
    r = DetectionResult(score=0.0, is_hallucination=False, confidence=1.0, method="test")
    assert r.score == 0.0
    assert r.confidence == 1.0

    r2 = DetectionResult(score=1.0, is_hallucination=True, confidence=0.0, method="test")
    assert r2.score == 1.0


def test_detection_result_score_below_zero_raises() -> None:
    """score < 0 should raise ValueError."""
    with pytest.raises(ValueError, match="score must be in"):
        DetectionResult(score=-0.01, is_hallucination=False, confidence=0.5, method="test")


def test_detection_result_score_above_one_raises() -> None:
    """score > 1 should raise ValueError."""
    with pytest.raises(ValueError, match="score must be in"):
        DetectionResult(score=1.01, is_hallucination=True, confidence=0.5, method="test")


def test_detection_result_confidence_below_zero_raises() -> None:
    """confidence < 0 should raise ValueError."""
    with pytest.raises(ValueError, match="confidence must be in"):
        DetectionResult(score=0.5, is_hallucination=False, confidence=-0.1, method="test")


def test_detection_result_confidence_above_one_raises() -> None:
    """confidence > 1 should raise ValueError."""
    with pytest.raises(ValueError, match="confidence must be in"):
        DetectionResult(score=0.5, is_hallucination=True, confidence=1.1, method="test")


def test_detection_result_details_default_empty() -> None:
    """details field defaults to an empty dict."""
    r = DetectionResult(score=0.5, is_hallucination=True, confidence=0.5, method="m")
    assert r.details == {}


def test_detection_result_details_stored() -> None:
    """Custom details are preserved."""
    r = DetectionResult(
        score=0.5,
        is_hallucination=True,
        confidence=0.5,
        method="m",
        details={"key": "value"},
    )
    assert r.details["key"] == "value"


# ---------------------------------------------------------------------------
# AggregatedResult validation
# ---------------------------------------------------------------------------


def test_aggregated_result_valid() -> None:
    """AggregatedResult accepts valid score."""
    r = AggregatedResult(score=0.7, is_hallucination=True, confidence=0.8)
    assert r.score == 0.7


def test_aggregated_result_score_below_zero_raises() -> None:
    """AggregatedResult rejects score < 0."""
    with pytest.raises(ValueError, match="score must be in"):
        AggregatedResult(score=-0.01, is_hallucination=False, confidence=0.5)


def test_aggregated_result_score_above_one_raises() -> None:
    """AggregatedResult rejects score > 1."""
    with pytest.raises(ValueError, match="score must be in"):
        AggregatedResult(score=1.5, is_hallucination=True, confidence=0.5)


def test_aggregated_result_defaults() -> None:
    """results and weights lists default to empty."""
    r = AggregatedResult(score=0.5, is_hallucination=False, confidence=0.5)
    assert r.results == []
    assert r.weights == []
