"""Example 2: Batch hallucination detection with parallel processing.

Demonstrates:
- Using detect_batch() for multiple claim/context pairs
- Parallel execution via ThreadPoolExecutor
- Summarising results across a batch
"""

from __future__ import annotations

import time

from groundtruth import GroundTruthDetector

# Use lightweight detector for this demo (no external models needed)
detector = GroundTruthDetector(threshold=0.5)

# ---------------------------------------------------------------------------
# Batch of (claim, context) pairs — mix of grounded and hallucinated
# ---------------------------------------------------------------------------
claims = [
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Amazon river flows through Asia.",
    "Dogs are warm-blooded mammals.",
    "Shakespeare wrote Romeo and Juliet in 1599.",
    "Penguins are found primarily in the Arctic near the North Pole.",
    "Python was created by Guido van Rossum.",
    "The Great Wall of China is visible from space with the naked eye.",
    "Light travels at approximately 299,792 kilometers per second.",
]

contexts = [
    "Water transitions from liquid to gas at 100°C (212°F) at sea level.",
    "The Amazon river is the largest river by discharge in South America.",
    "Dogs (Canis lupus familiaris) are domesticated mammals of the family Canidae.",
    "Romeo and Juliet is a tragedy written by William Shakespeare around 1594–1596.",
    "Penguins are flightless birds found exclusively in the Southern Hemisphere, particularly Antarctica.",
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "The Great Wall of China is not visible from space with the naked eye without aid.",
    "The speed of light in vacuum is exactly 299,792,458 metres per second.",
]

# ---------------------------------------------------------------------------
# Time the batch detection
# ---------------------------------------------------------------------------
start = time.perf_counter()
results = detector.detect_batch(claims, contexts)
elapsed = time.perf_counter() - start

print(f"Batch of {len(claims)} pairs processed in {elapsed * 1000:.1f}ms\n")
print(f"{'#':<3} {'Hallucinated?':<15} {'Score':<8} {'Claim (truncated)'}")
print("-" * 70)

hallucination_count = 0
for i, (claim, result) in enumerate(zip(claims, results), 1):
    marker = "YES" if result.is_hallucination else "no"
    if result.is_hallucination:
        hallucination_count += 1
    print(f"{i:<3} {marker:<15} {result.score:.3f}    {claim[:50]}...")

print("-" * 70)
print(f"\nTotal hallucinations detected: {hallucination_count}/{len(claims)}")
print(f"Average score: {sum(r.score for r in results) / len(results):.3f}")
