from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import DetectionResult


class BaseDetector(ABC):
    """Abstract base class for hallucination detectors."""

    @staticmethod
    def _validate_inputs(claim: str, context: str) -> None:
        """Validate that claim and context are strings.

        Args:
            claim: The generated text to evaluate.
            context: The ground-truth reference or source context.

        Raises:
            TypeError: If claim or context are not strings.
        """
        if not isinstance(claim, str):
            raise TypeError(f"claim must be a str, got {type(claim).__name__!r}")
        if not isinstance(context, str):
            raise TypeError(f"context must be a str, got {type(context).__name__!r}")

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

        Raises:
            TypeError: If claim or context are not strings.
        """
