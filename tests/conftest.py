"""Shared fixtures and mock helpers for the ground-truth test suite."""
from __future__ import annotations

import pytest

from groundtruth import (
    AggregatedResult,
    ConsistencyDetector,
    DetectionResult,
    EntailmentDetector,
    GroundTruthDetector,
    NLIProvider,
    OverlapDetector,
)


# ---------------------------------------------------------------------------
# NLI stub helpers
# ---------------------------------------------------------------------------


class FixedNLI:
    """NLIProvider stub that always returns a fixed entailment score."""

    def __init__(self, score: float) -> None:
        self._score = float(score)

    def entailment_score(self, premise: str, hypothesis: str) -> float:  # noqa: ARG002
        return self._score


class ClampingNLI:
    """NLIProvider stub that returns values outside [0,1] to test clamping."""

    def entailment_score(self, premise: str, hypothesis: str) -> float:  # noqa: ARG002
        return 1.5  # out-of-range; EntailmentDetector must clamp


# ---------------------------------------------------------------------------
# Detector fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def overlap_detector() -> OverlapDetector:
    return OverlapDetector(threshold=0.5)


@pytest.fixture()
def entailment_detector() -> EntailmentDetector:
    return EntailmentDetector(threshold=0.5)


@pytest.fixture()
def consistency_detector() -> ConsistencyDetector:
    return ConsistencyDetector(threshold=0.3)


@pytest.fixture()
def aggregator() -> GroundTruthDetector:
    return GroundTruthDetector(threshold=0.5)


@pytest.fixture()
def single_overlap_aggregator() -> GroundTruthDetector:
    return GroundTruthDetector(detectors=[OverlapDetector()], threshold=0.5)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grounded_pair() -> tuple[str, str]:
    """A claim that is genuinely supported by the context."""
    return (
        "The Eiffel Tower is in Paris France",
        "The Eiffel Tower is a famous landmark located in Paris, France.",
    )


@pytest.fixture()
def hallucinated_pair() -> tuple[str, str]:
    """A claim that contradicts / has no support in the context."""
    return (
        "The Eiffel Tower is located in Berlin Germany",
        "The Eiffel Tower is a famous landmark located in Paris, France.",
    )


@pytest.fixture()
def empty_claim_pair() -> tuple[str, str]:
    return ("", "Some reference text that exists in the context.")


@pytest.fixture()
def identical_pair() -> tuple[str, str]:
    text = "machine learning models require large amounts of training data"
    return (text, text)
