"""Tests for EntailmentDetector."""
import pytest

from groundtruth import EntailmentDetector, NLIProvider

# ---------------------------------------------------------------------------
# Minimal stub NLI provider for unit tests
# ---------------------------------------------------------------------------


class _FixedNLI:
    """Returns a fixed entailment score — useful for isolating detector logic."""

    def __init__(self, score: float) -> None:
        self._score = score

    def entailment_score(self, premise: str, hypothesis: str) -> float:  # noqa: ARG002
        return self._score


@pytest.fixture()
def detector() -> EntailmentDetector:
    return EntailmentDetector(threshold=0.5)


# ---------------------------------------------------------------------------
# Heuristic path
# ---------------------------------------------------------------------------


def test_entailment_fully_supported_claim(detector: EntailmentDetector) -> None:
    """Claim whose key terms all appear in context → low hallucination score."""
    result = detector.detect(
        claim="Paris is the capital of France",
        context="Paris is a major city and the capital of France located in Europe",
    )

    assert result.score < 0.4
    assert not result.is_hallucination


def test_entailment_unsupported_claim_is_hallucination(detector: EntailmentDetector) -> None:
    """Claim with key terms absent from context → hallucination flagged."""
    result = detector.detect(
        claim="The Amazon river flows through Egypt",
        context="The Nile river is the longest river in Africa",
    )

    assert result.is_hallucination
    assert result.score > 0.5


def test_entailment_partial_support_intermediate_score(detector: EntailmentDetector) -> None:
    """Partial term coverage → score between 0 and 1."""
    # "amazon" and "river" are in context, "egypt" is not → ~2/3 coverage
    result = detector.detect(
        claim="The Amazon river flows through Egypt",
        context="The Amazon river is the largest river in South America",
    )

    assert 0.0 < result.score < 1.0


def test_entailment_empty_claim_is_vacuously_true(detector: EntailmentDetector) -> None:
    """Empty claim has no key terms; treated as vacuously entailed."""
    result = detector.detect(claim="", context="any reference text here")

    assert not result.is_hallucination
    assert result.details["entailment_score"] == 1.0


def test_entailment_custom_nli_provider_used() -> None:
    """NLIProvider protocol is honoured; custom scores flow through correctly."""
    stub = _FixedNLI(score=0.9)
    det = EntailmentDetector(threshold=0.5, nli_provider=stub)
    result = det.detect(claim="anything", context="anything")

    assert result.details["entailment_score"] == pytest.approx(0.9)
    assert not result.is_hallucination


def test_entailment_nli_provider_low_score_flags_hallucination() -> None:
    """NLI provider returning low score → hallucination flagged."""
    stub = _FixedNLI(score=0.1)
    det = EntailmentDetector(threshold=0.5, nli_provider=stub)
    result = det.detect(claim="anything", context="anything")

    assert result.is_hallucination
    assert result.score == pytest.approx(0.9)


def test_entailment_protocol_satisfied_by_stub() -> None:
    """_FixedNLI satisfies the NLIProvider structural protocol."""
    assert isinstance(_FixedNLI(0.5), NLIProvider)
