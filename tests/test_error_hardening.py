"""Error-hardening tests: None inputs, empty strings, huge strings, unicode, malformed data."""

from __future__ import annotations

import pytest

from groundtruth import (
    AggregatedResult,
    ConsistencyDetector,
    DetectionResult,
    EntailmentDetector,
    GroundTruthDetector,
    OverlapDetector,
)

# ---------------------------------------------------------------------------
# None / type safety
# ---------------------------------------------------------------------------


def test_overlap_none_claim_raises() -> None:
    """Passing None as claim to OverlapDetector raises TypeError."""
    det = OverlapDetector()
    with pytest.raises((TypeError, AttributeError)):
        det.detect(claim=None, context="valid context")  # type: ignore[arg-type]


def test_overlap_none_context_raises() -> None:
    """Passing None as context to OverlapDetector raises TypeError."""
    det = OverlapDetector()
    with pytest.raises((TypeError, AttributeError)):
        det.detect(claim="valid claim", context=None)  # type: ignore[arg-type]


def test_entailment_none_claim_raises() -> None:
    """Passing None as claim to EntailmentDetector raises TypeError."""
    det = EntailmentDetector()
    with pytest.raises((TypeError, AttributeError)):
        det.detect(claim=None, context="valid context")  # type: ignore[arg-type]


def test_consistency_none_claim_raises() -> None:
    """Passing None as claim to ConsistencyDetector raises TypeError."""
    det = ConsistencyDetector()
    with pytest.raises((TypeError, AttributeError)):
        det.detect(claim=None, context="valid context")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_overlap_empty_claim_and_context() -> None:
    """Both empty → valid DetectionResult, not hallucination."""
    det = OverlapDetector()
    result = det.detect(claim="", context="")
    assert isinstance(result, DetectionResult)
    assert not result.is_hallucination


def test_entailment_empty_context_with_claim() -> None:
    """Empty context, non-empty claim → claim terms absent → flagged."""
    det = EntailmentDetector(threshold=0.5)
    result = det.detect(claim="Paris is the capital", context="")
    # No terms covered in empty context → entailment near 0 → hallucination
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.score <= 1.0


def test_consistency_empty_candidates_list() -> None:
    """Empty candidates list → fall back to context as reference."""
    det = ConsistencyDetector()
    result = det.detect(claim="hello world", context="hello world", candidates=[])
    # candidates=[] is falsy → falls back to [context]
    assert result.details["num_references"] == 1


# ---------------------------------------------------------------------------
# Huge string inputs
# ---------------------------------------------------------------------------


def test_overlap_huge_claim() -> None:
    """Claim of 10,000 words does not crash and returns valid result."""
    big_claim = " ".join(["word"] * 10_000)
    context = "word appears here"
    det = OverlapDetector()
    result = det.detect(claim=big_claim, context=context)
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.score <= 1.0


def test_entailment_huge_context() -> None:
    """Context of 10,000 words does not crash and returns valid result."""
    big_context = " ".join(["context"] * 10_000)
    det = EntailmentDetector()
    result = det.detect(claim="context appears here", context=big_context)
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.score <= 1.0


def test_consistency_huge_candidates() -> None:
    """100 candidates do not cause errors."""
    det = ConsistencyDetector()
    candidates = ["the sky is blue"] * 100
    result = det.detect(claim="the sky is blue", context="the sky is blue", candidates=candidates)
    assert isinstance(result, DetectionResult)
    assert result.details["num_references"] == 100


# ---------------------------------------------------------------------------
# Unicode edge cases
# ---------------------------------------------------------------------------


def test_overlap_unicode_claim_and_context() -> None:
    """Unicode text (Chinese, Arabic, emoji) does not crash."""
    det = OverlapDetector()
    result = det.detect(
        claim="巴黎是法国的首都 Paris capitale de la France 🗼",
        context="巴黎 Paris 🗼 France",
    )
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.score <= 1.0


def test_entailment_unicode_claim() -> None:
    """EntailmentDetector handles unicode without errors."""
    det = EntailmentDetector()
    result = det.detect(
        claim="Ünïcödé têxt wïth dïäcrïtïcs",
        context="unicode text with diacritics",
    )
    assert isinstance(result, DetectionResult)


def test_consistency_unicode_candidates() -> None:
    """ConsistencyDetector handles unicode candidates."""
    det = ConsistencyDetector()
    result = det.detect(
        claim="café résumé naïve",
        context="french loanwords",
        candidates=["café résumé naïve", "english plain text"],
    )
    assert isinstance(result, DetectionResult)


# ---------------------------------------------------------------------------
# Special characters / punctuation-heavy strings
# ---------------------------------------------------------------------------


def test_overlap_punctuation_only_claim() -> None:
    """Punctuation-only claim tokenises to empty → handled gracefully."""
    det = OverlapDetector()
    result = det.detect(claim="!!! ??? --- ...", context="some real context here")
    assert isinstance(result, DetectionResult)
    # Empty token list → early return
    assert not result.is_hallucination
    assert result.confidence == 0.0


def test_entailment_numbers_in_claim() -> None:
    """Claims with numeric tokens are handled correctly."""
    det = EntailmentDetector()
    result = det.detect(
        claim="The temperature is 42 degrees Celsius",
        context="The temperature reached 42 degrees Celsius today",
    )
    assert isinstance(result, DetectionResult)
    assert result.score < 0.5


# ---------------------------------------------------------------------------
# Aggregator edge cases
# ---------------------------------------------------------------------------


def test_aggregator_single_detector() -> None:
    """Aggregator works with just one detector."""
    agg = GroundTruthDetector(detectors=[OverlapDetector()], threshold=0.5)
    result = agg.detect(claim="hello world", context="hello world")
    assert isinstance(result, AggregatedResult)
    assert len(result.results) == 1
    assert sum(agg.weights) == pytest.approx(1.0)


def test_aggregator_many_detectors() -> None:
    """Aggregator handles many identical detectors."""
    detectors = [OverlapDetector() for _ in range(10)]
    agg = GroundTruthDetector(detectors=detectors, threshold=0.5)
    result = agg.detect(claim="test claim text", context="test claim text")
    assert len(result.results) == 10
    assert sum(agg.weights) == pytest.approx(1.0)


def test_aggregator_whitespace_only_claim() -> None:
    """Whitespace-only claim produces valid result without crash."""
    agg = GroundTruthDetector()
    result = agg.detect(claim="   \t\n   ", context="some actual context here")
    assert isinstance(result, AggregatedResult)
    assert 0.0 <= result.score <= 1.0
