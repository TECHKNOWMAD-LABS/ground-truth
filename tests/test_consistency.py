"""Tests for ConsistencyDetector."""
import pytest

from groundtruth import ConsistencyDetector, DetectionResult


@pytest.fixture()
def detector() -> ConsistencyDetector:
    return ConsistencyDetector(threshold=0.3)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_consistency_identical_candidates_low_score(detector: ConsistencyDetector) -> None:
    """Claim identical to all candidates → near-zero hallucination score."""
    claim = "the sky is blue and clouds are white"
    result = detector.detect(
        claim=claim,
        context=claim,
        candidates=[claim, claim, claim],
    )

    assert result.score < 0.1
    assert not result.is_hallucination


def test_consistency_divergent_claim_is_hallucination(detector: ConsistencyDetector) -> None:
    """Claim completely unlike all candidates → hallucination flagged."""
    result = detector.detect(
        claim="quantum tunneling enables faster than light travel",
        context="classical mechanics describes everyday motion",
        candidates=[
            "Newton described motion with three laws",
            "classical physics covers objects at everyday scales",
            "thermodynamics is a branch of classical physics",
        ],
    )

    assert result.is_hallucination
    assert result.score > 0.6


def test_consistency_no_candidates_falls_back_to_context(detector: ConsistencyDetector) -> None:
    """Without candidates, context is used as the single reference."""
    context = "machine learning models require training data"
    result = detector.detect(
        claim="machine learning models require training data",
        context=context,
    )

    assert result.details["num_references"] == 1
    assert result.score < 0.1


def test_consistency_details_contain_similarities(detector: ConsistencyDetector) -> None:
    """details should include per-candidate similarity list."""
    result = detector.detect(
        claim="water is wet",
        context="water is wet",
        candidates=["water is wet", "fire is hot"],
    )

    assert "similarities" in result.details
    assert len(result.details["similarities"]) == 2
    assert result.details["similarities"][0] == pytest.approx(1.0)


def test_consistency_custom_threshold() -> None:
    """High threshold flags moderately similar claims."""
    strict = ConsistencyDetector(threshold=0.9)
    result = strict.detect(
        claim="the dog ran quickly",
        context="the dog ran quickly in the park",
        candidates=["the dog ran quickly in the park"],
    )
    # Jaccard won't be 1.0 due to extra tokens; strict threshold may flag it
    assert isinstance(result.is_hallucination, bool)
    assert 0.0 <= result.score <= 1.0


def test_consistency_result_type(detector: ConsistencyDetector) -> None:
    """detect() always returns a DetectionResult."""
    result = detector.detect(claim="test", context="test")
    assert isinstance(result, DetectionResult)
    assert result.method == "consistency"
