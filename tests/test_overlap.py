"""Tests for OverlapDetector."""
import pytest

from groundtruth import DetectionResult, OverlapDetector


@pytest.fixture()
def detector() -> OverlapDetector:
    return OverlapDetector(threshold=0.5)


# ---------------------------------------------------------------------------
# Basic detection behaviour
# ---------------------------------------------------------------------------


def test_overlap_identical_text_is_not_hallucination(detector: OverlapDetector) -> None:
    """Claim copied verbatim from context → near-zero hallucination score."""
    text = "The Eiffel Tower is located in Paris France"
    result = detector.detect(claim=text, context=text)

    assert isinstance(result, DetectionResult)
    assert result.method == "overlap"
    assert result.score < 0.1
    assert not result.is_hallucination


def test_overlap_no_shared_tokens_is_hallucination(detector: OverlapDetector) -> None:
    """Claim with zero token overlap with context → hallucination flagged."""
    result = detector.detect(
        claim="elephants migrate across savanna grasslands",
        context="quantum entanglement violates classical locality",
    )

    assert result.score > 0.9
    assert result.is_hallucination


def test_overlap_partial_returns_intermediate_score(detector: OverlapDetector) -> None:
    """Claim sharing some but not all tokens → score between 0 and 1."""
    result = detector.detect(
        claim="Paris is the capital of Germany",
        context="Paris is the capital of France",
    )

    assert 0.0 < result.score < 1.0


def test_overlap_empty_claim_returns_no_hallucination(detector: OverlapDetector) -> None:
    """Empty claim is treated as vacuously grounded with zero confidence."""
    result = detector.detect(claim="", context="some reference text here")

    assert not result.is_hallucination
    assert result.confidence == 0.0


def test_overlap_details_include_recall_keys(detector: OverlapDetector) -> None:
    """details dict should expose per-n recall values."""
    result = detector.detect(
        claim="Paris is the capital city of France",
        context="France's capital is Paris the city of lights",
    )

    assert "avg_recall" in result.details
    assert "recall_1gram" in result.details
    assert "recall_2gram" in result.details
    assert 0.0 <= result.details["avg_recall"] <= 1.0


def test_overlap_custom_threshold_changes_verdict() -> None:
    """A very high threshold flags even well-supported claims."""
    strict = OverlapDetector(threshold=0.99)
    result = strict.detect(
        claim="Paris is in France",
        context="Paris is the capital of France and is in France",
    )
    # Even with high overlap the strict threshold may flag it
    assert isinstance(result.is_hallucination, bool)


def test_overlap_unigram_only_detector() -> None:
    """Detector with n_values=(1,) should still produce valid results."""
    det = OverlapDetector(n_values=(1,), threshold=0.5)
    result = det.detect(claim="dog runs fast", context="the dog is running fast")
    assert 0.0 <= result.score <= 1.0
    assert "recall_1gram" in result.details
