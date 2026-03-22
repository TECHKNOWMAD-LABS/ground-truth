from __future__ import annotations

from typing import Any

from .base import BaseDetector
from .detectors.consistency import ConsistencyDetector
from .detectors.entailment import EntailmentDetector
from .detectors.overlap import OverlapDetector
from .models import AggregatedResult


class GroundTruthDetector:
    """Aggregate multiple hallucination detectors into a single weighted score.

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
    ) -> None:
        """
        Args:
            detectors: Detector instances to run.  Defaults to
                       [OverlapDetector, EntailmentDetector, ConsistencyDetector].
            weights: Per-detector weights (need not sum to 1; normalised internally).
                     Defaults to uniform weights.
            threshold: Aggregated score above which a claim is flagged as a
                       hallucination.
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
        """
        results = [d.detect(claim, context, **kwargs) for d in self.detectors]

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

    def detect_batch(
        self,
        claims: list[str],
        contexts: list[str],
        **kwargs: Any,
    ) -> list[AggregatedResult]:
        """Run detection on multiple (claim, context) pairs.

        Args:
            claims: List of generated texts to evaluate.
            contexts: Corresponding list of reference contexts.

        Returns:
            List of AggregatedResult in the same order as inputs.
        """
        if len(claims) != len(contexts):
            raise ValueError(
                f"claims and contexts must have the same length: "
                f"{len(claims)} vs {len(contexts)}"
            )
        return [self.detect(claim, ctx, **kwargs) for claim, ctx in zip(claims, contexts)]

    def add_detector(self, detector: BaseDetector, weight: float = 1.0) -> None:
        """Append a detector and re-normalise weights.

        Args:
            detector: New detector to add.
            weight: Unnormalised weight for the new detector relative to existing ones.
        """
        if weight <= 0:
            raise ValueError("weight must be positive")
        # De-normalise existing weights, append new, re-normalise
        raw = [w * len(self.weights) for w in self.weights] + [weight]
        total = sum(raw)
        self.detectors.append(detector)
        self.weights = [w / total for w in raw]
