from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import DetectionResult


class BaseDetector(ABC):
    """Abstract base class for hallucination detectors."""

    @abstractmethod
    def detect(self, claim: str, context: str, **kwargs: Any) -> DetectionResult:
        """
        Detect whether a claim is hallucinated relative to the context.

        Args:
            claim: The generated text to evaluate.
            context: The ground-truth reference or source context.
            **kwargs: Detector-specific arguments.

        Returns:
            DetectionResult with hallucination score and metadata.
        """
