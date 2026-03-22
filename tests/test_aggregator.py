"""Tests for GroundTruthDetector aggregator."""
import pytest

from groundtruth import (
    AggregatedResult,
    ConsistencyDetector,
    DetectionResult,
    EntailmentDetector,
    GroundTruthDetector,
    OverlapDetector,
)


@pytest.fixture()
def aggregator() -> GroundTruthDetector:
    return GroundTruthDetector(threshold=0.5)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def test_aggregator_returns_aggregated_result(aggregator: GroundTruthDetector) -> None:
    """detect() returns an AggregatedResult with one sub-result per detector."""
    result = aggregator.detect(
        claim="Paris is the capital of France",
        context="France's capital city is Paris located in Western Europe",
    )

    assert isinstance(result, AggregatedResult)
    assert len(result.results) == 3  # default: overlap, entailment, consistency
    assert all(isinstance(r, DetectionResult) for r in result.results)


def test_aggregator_score_in_unit_interval(aggregator: GroundTruthDetector) -> None:
    """Aggregated score must always be in [0, 1]."""
    result = aggregator.detect(claim="random text", context="completely different content")
    assert 0.0 <= result.score <= 1.0


def test_aggregator_weights_sum_to_one(aggregator: GroundTruthDetector) -> None:
    """Internal weights should be normalised to sum to 1.0."""
    assert sum(aggregator.weights) == pytest.approx(1.0)


def test_aggregator_custom_weights_normalised() -> None:
    """Custom weights are normalised regardless of their raw sum."""
    det = GroundTruthDetector(
        detectors=[OverlapDetector(), EntailmentDetector()],
        weights=[3.0, 1.0],
    )
    assert sum(det.weights) == pytest.approx(1.0)
    assert det.weights[0] == pytest.approx(0.75)
    assert det.weights[1] == pytest.approx(0.25)


def test_aggregator_mismatched_weights_raises() -> None:
    """Providing wrong number of weights raises ValueError."""
    with pytest.raises(ValueError, match="weights length"):
        GroundTruthDetector(
            detectors=[OverlapDetector(), EntailmentDetector()],
            weights=[1.0, 1.0, 1.0],  # 3 weights for 2 detectors
        )


def test_aggregator_batch_returns_list(aggregator: GroundTruthDetector) -> None:
    """detect_batch() returns one result per input pair."""
    claims = ["Paris is in France", "Dogs are mammals", "Ice is cold"]
    contexts = [
        "France's capital is Paris",
        "Dogs are warm-blooded animals classified as mammals",
        "Ice is frozen water and is cold",
    ]
    results = aggregator.detect_batch(claims, contexts)

    assert len(results) == 3
    assert all(isinstance(r, AggregatedResult) for r in results)


def test_aggregator_batch_length_mismatch_raises(aggregator: GroundTruthDetector) -> None:
    """detect_batch raises ValueError when claims/contexts differ in length."""
    with pytest.raises(ValueError, match="same length"):
        aggregator.detect_batch(["claim one", "claim two"], ["only one context"])


def test_aggregator_add_detector_renormalises_weights(aggregator: GroundTruthDetector) -> None:
    """add_detector appends detector and re-normalises weights."""
    initial_count = len(aggregator.detectors)
    aggregator.add_detector(ConsistencyDetector(), weight=1.0)

    assert len(aggregator.detectors) == initial_count + 1
    assert sum(aggregator.weights) == pytest.approx(1.0)


def test_aggregator_grounded_claim_low_score() -> None:
    """Well-supported claim should receive a low aggregated hallucination score."""
    det = GroundTruthDetector()
    result = det.detect(
        claim="The Eiffel Tower is in Paris",
        context="The Eiffel Tower is a famous landmark located in Paris France",
    )

    assert result.score < 0.5
    assert not result.is_hallucination


def test_aggregator_hallucinated_claim_high_score() -> None:
    """Claim contradicting context should receive a high aggregated score."""
    det = GroundTruthDetector()
    result = det.detect(
        claim="penguins live in the Arctic near polar bears",
        context="penguins are found exclusively in the Southern Hemisphere including Antarctica",
    )

    assert result.score > 0.3  # some hallucination signal detected


def test_aggregator_candidates_forwarded_to_consistency() -> None:
    """kwargs such as 'candidates' are forwarded to all detectors."""
    det = GroundTruthDetector(detectors=[ConsistencyDetector()])
    candidates = ["dogs are mammals", "dogs are warm blooded"]
    result = det.detect(
        claim="dogs are mammals",
        context="dogs are mammals",
        candidates=candidates,
    )
    # ConsistencyDetector uses the candidates; we get a valid result back
    assert isinstance(result, AggregatedResult)
    assert result.results[0].details["num_references"] == len(candidates)
