"""GroundTruth — hallucination detector for LLM outputs."""

from .aggregator import GroundTruthDetector
from .base import BaseDetector
from .detectors.consistency import ConsistencyDetector
from .detectors.entailment import EntailmentDetector, NLIProvider
from .detectors.overlap import OverlapDetector
from .models import AggregatedResult, DetectionResult

__all__ = [
    "AggregatedResult",
    "BaseDetector",
    "ConsistencyDetector",
    "DetectionResult",
    "EntailmentDetector",
    "GroundTruthDetector",
    "NLIProvider",
    "OverlapDetector",
]
