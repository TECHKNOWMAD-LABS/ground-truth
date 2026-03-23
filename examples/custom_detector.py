"""Example 3: Custom detector composition and NLI provider integration.

Demonstrates:
- Using individual detectors (OverlapDetector, EntailmentDetector, ConsistencyDetector)
- Custom weighting: overlap 50%, entailment 30%, consistency 20%
- Plugging in a custom NLIProvider (stub simulating a real model)
- Using add_detector() to extend an existing aggregator at runtime
- ConsistencyDetector with multiple sampled model responses
"""
from __future__ import annotations

from groundtruth import (
    AggregatedResult,
    ConsistencyDetector,
    DetectionResult,
    EntailmentDetector,
    GroundTruthDetector,
    NLIProvider,
    OverlapDetector,
)


# ---------------------------------------------------------------------------
# Custom NLI provider (stub — replace with real HuggingFace model in production)
# ---------------------------------------------------------------------------


class SimpleKeywordNLI:
    """Simple keyword-overlap NLI stub for demonstration.

    In production, replace with a real cross-encoder NLI model:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    """

    def entailment_score(self, premise: str, hypothesis: str) -> float:
        """Estimate entailment using keyword overlap (production: use real NLI)."""
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())
        if not hypothesis_words:
            return 1.0
        overlap = len(premise_words & hypothesis_words)
        return min(overlap / len(hypothesis_words), 1.0)


# ---------------------------------------------------------------------------
# Example 1: Custom weighted aggregator
# ---------------------------------------------------------------------------
print("=== Custom Weighted Aggregator ===\n")

nli = SimpleKeywordNLI()

detector = GroundTruthDetector(
    detectors=[
        OverlapDetector(n_values=(1, 2), threshold=0.4),
        EntailmentDetector(threshold=0.5, nli_provider=nli),
        ConsistencyDetector(threshold=0.3),
    ],
    weights=[0.5, 0.3, 0.2],  # overlap most important, consistency least
    threshold=0.45,
)

claim = "Climate change is primarily caused by human activity."
context = (
    "Scientific consensus shows that human activities, especially burning fossil fuels, "
    "are the dominant cause of observed climate change since the mid-20th century."
)

result = detector.detect(claim=claim, context=context)
print(f"Claim: {claim}")
print(f"Score: {result.score:.3f}  |  Hallucination: {result.is_hallucination}")
for r, w in zip(result.results, result.weights):
    print(f"  [{r.method:12s}] weight={w:.2f}  score={r.score:.3f}")

print()

# ---------------------------------------------------------------------------
# Example 2: Self-consistency detection with multiple candidates
# ---------------------------------------------------------------------------
print("=== Self-Consistency Detection ===\n")

cons_detector = ConsistencyDetector(threshold=0.3)

# Simulated sampled responses from an LLM (in production, sample multiple times)
candidates = [
    "The Earth orbits the Sun once every 365.25 days.",
    "Earth completes one revolution around the Sun in approximately 365 days.",
    "The Earth takes one year, about 365 days, to orbit the Sun.",
    "Earth's orbital period around the Sun is 365.25 days.",
]

claim_consistent = "The Earth completes one orbit around the Sun every 365 days."
claim_inconsistent = "The Earth orbits the Sun every 30 days like the Moon."

r1 = cons_detector.detect(claim=claim_consistent, context="", candidates=candidates)
r2 = cons_detector.detect(claim=claim_inconsistent, context="", candidates=candidates)

print(f"Consistent claim:   score={r1.score:.3f}  hallu={r1.is_hallucination}")
print(f"Inconsistent claim: score={r2.score:.3f}  hallu={r2.is_hallucination}")

print()

# ---------------------------------------------------------------------------
# Example 3: Adding a detector at runtime
# ---------------------------------------------------------------------------
print("=== Runtime Detector Addition ===\n")

base = GroundTruthDetector(
    detectors=[OverlapDetector()],
    threshold=0.5,
)
print(f"Before: {len(base.detectors)} detector(s), weights={[f'{w:.2f}' for w in base.weights]}")

base.add_detector(EntailmentDetector(), weight=1.0)
print(f"After:  {len(base.detectors)} detector(s), weights={[f'{w:.2f}' for w in base.weights]}")

result = base.detect(
    claim="The Moon is Earth's only natural satellite.",
    context="The Moon is a natural satellite of Earth and the fifth-largest satellite in the Solar System.",
)
print(f"Result: score={result.score:.3f}  is_hallucination={result.is_hallucination}")
