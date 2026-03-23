"""Example 1: Basic hallucination detection with GroundTruthDetector.

Demonstrates:
- Creating a GroundTruthDetector with default detectors
- Detecting a grounded claim vs. a hallucinated claim
- Reading score, is_hallucination, and per-detector results
"""
from __future__ import annotations

from groundtruth import GroundTruthDetector

# Create detector with default 3-detector ensemble
detector = GroundTruthDetector(threshold=0.5)

# ---------------------------------------------------------------------------
# Example 1: Grounded claim — should NOT be flagged
# ---------------------------------------------------------------------------
grounded_result = detector.detect(
    claim="The Eiffel Tower is located in Paris, France.",
    context=(
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
        "France. It is named after the engineer Gustave Eiffel."
    ),
)

print("=== Grounded Claim ===")
print(f"Claim:           The Eiffel Tower is located in Paris, France.")
print(f"Score:           {grounded_result.score:.3f}  (0.0=grounded, 1.0=hallucinated)")
print(f"Is hallucination: {grounded_result.is_hallucination}")
print(f"Confidence:      {grounded_result.confidence:.3f}")
print()
print("Per-detector breakdown:")
for r in grounded_result.results:
    print(f"  [{r.method:12s}] score={r.score:.3f}  hallu={r.is_hallucination}")

print()

# ---------------------------------------------------------------------------
# Example 2: Hallucinated claim — should BE flagged
# ---------------------------------------------------------------------------
hallucinated_result = detector.detect(
    claim="The Eiffel Tower is located in Berlin, Germany.",
    context=(
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
        "France. It is named after the engineer Gustave Eiffel."
    ),
)

print("=== Hallucinated Claim ===")
print(f"Claim:           The Eiffel Tower is located in Berlin, Germany.")
print(f"Score:           {hallucinated_result.score:.3f}  (0.0=grounded, 1.0=hallucinated)")
print(f"Is hallucination: {hallucinated_result.is_hallucination}")
print(f"Confidence:      {hallucinated_result.confidence:.3f}")
print()
print("Per-detector breakdown:")
for r in hallucinated_result.results:
    print(f"  [{r.method:12s}] score={r.score:.3f}  hallu={r.is_hallucination}")
