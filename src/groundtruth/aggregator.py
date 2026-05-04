from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

from .base import BaseDetector
from .detectors.consistency import ConsistencyDetector
from .detectors.entailment import EntailmentDetector
from .detectors.overlap import OverlapDetector
from .models import AggregatedResult, DetectionResult


class GroundTruthDetector:
    """Aggregate multiple hallucination detectors into a single weighted score.

    Detectors are run in parallel via ``ThreadPoolExecutor`` for batch
    workloads, yielding near-linear speedup when detectors are I/O-bound
    (e.g. NLI model inference).

    Example::

        detector = GroundTruthDetector()
        result = detector.detect(
            claim="The Eiffel Tower is in Berlin.",
            context="The Eiffel Tower is a landmark in Paris, France.",
        )
        print(result.is_hallucination, result.score)
    """

    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
        weights: list[float] | None = None,
        threshold: float = 0.5,
        max_workers: int | None = None,
    ) -> None:
        """
        Args:
            detectors: Detector instances to run.  Defaults to
                       [OverlapDetector, EntailmentDetector, ConsistencyDetector].
            weights: Per-detector weights (need not sum to 1; normalised internally).
                     Defaults to uniform weights.
            threshold: Aggregated score above which a claim is flagged as a
                       hallucination.
            max_workers: Maximum threads for parallel batch processing.
                         Defaults to min(32, len(detectors) + 4).
        """
        if detectors is None:
            detectors = [OverlapDetector(), EntailmentDetector(), ConsistencyDetector()]

        if weights is not None and len(weights) != len(detectors):
            raise ValueError(
                f"weights length ({len(weights)}) must match detectors length ({len(detectors)})"
            )

        if weights is None:
            weights = [1.0] * len(detectors)

        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")

        self.detectors = list(detectors)
        self.weights = [w / total for w in weights]
        self.threshold = threshold
        self._max_workers = max_workers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(self, results: list[DetectionResult]) -> AggregatedResult:
        """Weighted aggregation of a list of DetectionResults."""
        weighted_score = sum(r.score * w for r, w in zip(results, self.weights))
        weighted_confidence = sum(r.confidence * w for r, w in zip(results, self.weights))

        # Clamp to [0, 1] after weighting
        weighted_score = max(0.0, min(1.0, weighted_score))

        return AggregatedResult(
            score=weighted_score,
            is_hallucination=weighted_score > self.threshold,
            confidence=min(weighted_confidence, 1.0),
            results=results,
            weights=list(self.weights),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, claim: str, context: str, **kwargs: Any) -> AggregatedResult:
        """Run all detectors on a single (claim, context) pair.

        Args:
            claim: The generated text to evaluate.
            context: Ground-truth reference or source context.
            **kwargs: Forwarded to each detector (e.g. ``candidates`` for
                      ConsistencyDetector).

        Returns:
            AggregatedResult with weighted score and per-detector results.

        Raises:
            TypeError: If claim or context are not strings.
        """
        # Sequential execution for single pairs — avoids thread overhead
        results = [d.detect(claim, context, **kwargs) for d in self.detectors]
        return self._aggregate(results)

    def detect_batch(
        self,
        claims: list[str],
        contexts: list[str],
        **kwargs: Any,
    ) -> list[AggregatedResult]:
        """Run detection on multiple (claim, context) pairs in parallel.

        Uses ``ThreadPoolExecutor`` to parallelise across pairs.
        Pairs are processed concurrently — order of results matches inputs.

        # Performance: before — O(N * D) sequential; after — O(D) per worker
        # ~Nx speedup for large batches where N = len(claims)

        Args:
            claims: List of generated texts to evaluate.
            contexts: Corresponding list of reference contexts.

        Returns:
            List of AggregatedResult in the same order as inputs.

        Raises:
            ValueError: If claims and contexts have different lengths.
            TypeError: If any element is not a string.
        """
        if len(claims) != len(contexts):
            raise ValueError(
                f"claims and contexts must have the same length: {len(claims)} vs {len(contexts)}"
            )
        if not all(isinstance(c, str) for c in claims):
            raise TypeError("All claims must be strings")
        if not all(isinstance(c, str) for c in contexts):
            raise TypeError("All contexts must be strings")

        if not claims:
            return []

        # Use parallel execution for batches > 1
        if len(claims) == 1:
            return [self.detect(claims[0], contexts[0], **kwargs)]

        workers = self._max_workers or min(32, len(claims) + 4)
        output: list[AggregatedResult | None] = [None] * len(claims)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self.detect, claim, ctx, **kwargs): idx
                for idx, (claim, ctx) in enumerate(zip(claims, contexts))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                output[idx] = future.result()

        return output  # type: ignore[return-value]

    def add_detector(self, detector: BaseDetector, weight: float = 1.0) -> None:
        """Append a detector and re-normalise weights.

        Args:
            detector: New detector to add.
            weight: Unnormalised weight for the new detector relative to existing ones.

        Raises:
            ValueError: If weight is not positive.
        """
        if weight <= 0:
            raise ValueError("weight must be positive")
        # De-normalise existing weights, append new, re-normalise
        raw = [w * len(self.weights) for w in self.weights] + [weight]
        total = sum(raw)
        self.detectors.append(detector)
        self.weights = [w / total for w in raw]


# ---------------------------------------------------------------------------
# Module-level cached tokenizer for frequently reused text preprocessing
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402


@lru_cache(maxsize=2048)
def cached_tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text with LRU cache — avoids repeated regex on identical inputs.

    Args:
        text: Input text to tokenize.

    Returns:
        Tuple of lowercase word tokens (hashable for caching).
    """
    return tuple(_re.findall(r"\b\w+\b", text.lower()))
