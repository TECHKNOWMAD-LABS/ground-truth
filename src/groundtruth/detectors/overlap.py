from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

from ..base import BaseDetector
from ..models import DetectionResult


@lru_cache(maxsize=1024)
def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens with LRU cache.

    Caching avoids repeated regex evaluation on identical strings, which is
    common in batch detection where the same context appears multiple times.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def _ngram_recall(claim_tokens: list[str], context_tokens: list[str], n: int) -> float:
    """Fraction of claim n-grams that appear in context (recall-oriented)."""
    claim_ng = Counter(tuple(claim_tokens[i : i + n]) for i in range(len(claim_tokens) - n + 1))
    context_ng = Counter(
        tuple(context_tokens[i : i + n]) for i in range(len(context_tokens) - n + 1)
    )
    if not claim_ng:
        return 0.0
    overlap = sum((claim_ng & context_ng).values())
    return overlap / sum(claim_ng.values())


class OverlapDetector(BaseDetector):
    """Detect hallucinations via n-gram recall of the claim against the context.

    A low recall indicates the claim contains terms not present in the context,
    suggesting a potential hallucination.
    """

    def __init__(self, n_values: tuple[int, ...] = (1, 2), threshold: float = 0.5) -> None:
        """
        Args:
            n_values: N-gram sizes to compute recall for.
            threshold: Minimum average recall to consider the claim grounded.
                       Claims below this threshold are flagged as hallucinations.
        """
        self.n_values = n_values
        self.threshold = threshold

    def detect(self, claim: str, context: str, **kwargs: Any) -> DetectionResult:
        self._validate_inputs(claim, context)
        claim_tokens = _tokenize(claim)
        context_tokens = _tokenize(context)

        if not claim_tokens:
            return DetectionResult(
                score=0.0, is_hallucination=False, confidence=0.0, method="overlap"
            )

        recalls: dict[str, float] = {}
        for n in self.n_values:
            if len(claim_tokens) >= n:
                recalls[f"recall_{n}gram"] = _ngram_recall(claim_tokens, context_tokens, n)

        avg_recall = sum(recalls.values()) / len(recalls) if recalls else 0.0
        hallucination_score = 1.0 - avg_recall
        is_hallucination = avg_recall < self.threshold

        dist = abs(avg_recall - self.threshold)
        max_dist = max(self.threshold, 1.0 - self.threshold)
        confidence = min(dist / max_dist, 1.0) if max_dist > 0 else 0.0

        return DetectionResult(
            score=hallucination_score,
            is_hallucination=is_hallucination,
            confidence=confidence,
            method="overlap",
            details={"avg_recall": avg_recall, **recalls},
        )
