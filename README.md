# GroundTruth

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

Hallucination detection for LLM outputs. Evaluate whether generated text remains grounded in source context using three complementary methods — overlap, entailment, and consistency — aggregated into a single calibrated score.

---

## Features

- **Three detection methods** — n-gram recall (overlap), semantic key-term coverage (entailment), and Jaccard similarity across candidates (consistency).
- **Weighted aggregation** — combine detectors with custom weights; weights are auto-normalized when detectors are added or removed.
- **Confidence scoring** — every result includes a confidence value (distance from decision boundary) alongside the hallucination score.
- **Pluggable NLI backend** — swap in any HuggingFace or custom NLI model via the `NLIProvider` protocol; heuristic fallback requires zero dependencies.
- **Batch processing** — `detect_batch` handles multiple claim/context pairs efficiently in a single call.
- **Zero required dependencies** — core runs on the standard library; `torch` and `transformers` are optional extras.

---

## Quick Start

```bash
pip install -e .
```

```python
from groundtruth import GroundTruthDetector, OverlapDetector, EntailmentDetector, ConsistencyDetector

detector = GroundTruthDetector(
    detectors=[
        (OverlapDetector(), 1.0),
        (EntailmentDetector(), 1.0),
        (ConsistencyDetector(), 0.5),
    ]
)

context = "The Eiffel Tower is located in Paris, France, and was completed in 1889."
claim   = "The Eiffel Tower is in London."

result = detector.detect(claim, context)
print(result.score)           # e.g. 0.83 (closer to 1.0 = likely hallucination)
print(result.is_hallucination) # True
print(result.confidence)      # e.g. 0.66
```

### Batch detection

```python
claims   = ["Paris is the capital of France.", "The tower was built in 1950."]
contexts = [context, context]

results = detector.detect_batch(claims, contexts)
for r in results:
    print(r.score, r.is_hallucination)
```

### Optional NLI backend

```bash
pip install -e ".[transformers]"
```

```python
from groundtruth import EntailmentDetector
from transformers import pipeline

nli = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")

class HFProvider:
    def predict(self, premise: str, hypothesis: str) -> float:
        out = nli(premise, candidate_labels=[hypothesis])
        return 1.0 - out["scores"][0]  # invert: entailed → low score

detector = EntailmentDetector(nli_provider=HFProvider())
```

---

## Architecture

```
src/groundtruth/
├── __init__.py        # Public API exports
├── base.py            # Abstract BaseDetector (detect → DetectionResult)
├── models.py          # DetectionResult, AggregatedResult dataclasses
├── aggregator.py      # GroundTruthDetector — orchestrates detectors, batching, weight normalization
└── detectors/
    ├── overlap.py     # N-gram recall: computes unigram/bigram coverage of claim against context
    ├── entailment.py  # Key-term heuristic or pluggable NLI backend via NLIProvider protocol
    └── consistency.py # Jaccard similarity across candidate responses; falls back to context
```

**Score semantics:** `0.0` = fully grounded, `1.0` = likely hallucination. Each detector normalizes independently; the aggregator computes a weighted average and re-normalizes weights on every `add_detector` call.

**Extending:** subclass `BaseDetector`, implement `detect(claim, context, **kwargs) -> DetectionResult`, and pass the instance to `GroundTruthDetector`.

---

## Development

```bash
pip install -e ".[dev]"
pytest -v          # run all tests
ruff check .       # lint
```

Tests live in `tests/` and cover core behavior, edge cases, weight normalization, batch processing, and the `NLIProvider` protocol.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the fork → feature branch → test → lint → PR workflow.

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by [TechKnowMad Labs](https://techknowmad.ai)
