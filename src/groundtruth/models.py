from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    """Result from a single hallucination detector."""

    score: float  # 0.0 = grounded, 1.0 = likely hallucination
    is_hallucination: bool
    confidence: float  # 0.0 = uncertain, 1.0 = certain
    method: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class AggregatedResult:
    """Weighted aggregate of multiple DetectionResults."""

    score: float  # 0.0 = grounded, 1.0 = likely hallucination
    is_hallucination: bool
    confidence: float
    results: list[DetectionResult] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")
