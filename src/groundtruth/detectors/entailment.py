from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from ..base import BaseDetector
from ..models import DetectionResult

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "and", "or", "but", "nor", "for", "yet", "so", "as", "at", "by",
        "from", "in", "into", "of", "on", "to", "up", "with", "about",
        "that", "this", "these", "those", "it", "its", "i", "you", "he", "she",
        "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "their", "our", "not", "no",
    }
)


def _key_terms(text: str) -> list[str]:
    """Extract content-bearing terms (strip stopwords and short tokens)."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


@runtime_checkable
class NLIProvider(Protocol):
    """Protocol for pluggable NLI backends (e.g., HuggingFace transformers)."""

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """Return P(premise entails hypothesis) in [0, 1]."""
        ...


class EntailmentDetector(BaseDetector):
    """Detect hallucinations by checking whether the context entails the claim.

    Uses a key-term coverage heuristic by default.  Drop in a custom
    ``NLIProvider`` to use a proper NLI model (e.g., cross-encoder/nli-*).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        nli_provider: NLIProvider | None = None,
    ) -> None:
        """
        Args:
            threshold: Minimum entailment score to consider the claim grounded.
            nli_provider: Optional NLI backend.  Defaults to heuristic key-term
                          coverage when not provided.
        """
        self.threshold = threshold
        self.nli_provider = nli_provider

    def detect(self, claim: str, context: str, **kwargs: Any) -> DetectionResult:
        if self.nli_provider is not None:
            entailment = float(self.nli_provider.entailment_score(context, claim))
        else:
            entailment = self._heuristic_entailment(claim, context)

        entailment = max(0.0, min(1.0, entailment))
        hallucination_score = 1.0 - entailment
        is_hallucination = entailment < self.threshold

        dist = abs(entailment - self.threshold)
        max_dist = max(self.threshold, 1.0 - self.threshold)
        confidence = min(dist / max_dist, 1.0) if max_dist > 0 else 0.0

        return DetectionResult(
            score=hallucination_score,
            is_hallucination=is_hallucination,
            confidence=confidence,
            method="entailment",
            details={"entailment_score": entailment},
        )

    def _heuristic_entailment(self, claim: str, context: str) -> float:
        """Soft entailment: fraction of claim key-terms covered by context."""
        claim_terms = _key_terms(claim)
        if not claim_terms:
            return 1.0  # Vacuously supported
        context_words = set(re.findall(r"\b\w+\b", context.lower()))
        covered = sum(1 for t in claim_terms if t in context_words)
        return covered / len(claim_terms)
