from __future__ import annotations

import re
from typing import Any

from ..base import BaseDetector
from ..models import DetectionResult


def _jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two texts."""
    tokens_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", text_b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class ConsistencyDetector(BaseDetector):
    """Detect hallucinations by measuring self-consistency across candidates.

    Compares the claim against multiple sampled model responses (candidates).
    A low average similarity indicates the claim diverges from the consensus,
    suggesting it may be hallucinated.

    When no candidates are provided, the context itself is used as the sole
    reference for a single-reference consistency check.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        """
        Args:
            threshold: Minimum average Jaccard similarity to consider the claim
                       consistent with the candidate pool.
        """
        self.threshold = threshold

    def detect(
        self,
        claim: str,
        context: str,
        candidates: list[str] | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        """
        Args:
            claim: The generated text to evaluate.
            context: Source context (used as fallback when no candidates given).
            candidates: Optional list of sampled model responses to compare against.
        """
        references = candidates if candidates else [context]

        similarities = [_jaccard(claim, ref) for ref in references]
        avg_similarity = sum(similarities) / len(similarities)

        hallucination_score = 1.0 - avg_similarity
        is_hallucination = avg_similarity < self.threshold

        dist = abs(avg_similarity - self.threshold)
        max_dist = max(self.threshold, 1.0 - self.threshold)
        confidence = min(dist / max_dist, 1.0) if max_dist > 0 else 0.0

        return DetectionResult(
            score=hallucination_score,
            is_hallucination=is_hallucination,
            confidence=confidence,
            method="consistency",
            details={
                "avg_similarity": avg_similarity,
                "num_references": len(references),
                "similarities": similarities,
            },
        )
