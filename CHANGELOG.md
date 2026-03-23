# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2026-03-23

Initial release following 8 Edgecraft autonomous iteration cycles.

### Added — Cycle 1: Test Coverage

- `tests/conftest.py`: Shared fixtures (`FixedNLI`, `ClampingNLI`, detector fixtures, sample data pairs)
- `tests/test_models.py`: 10 tests covering `DetectionResult` and `AggregatedResult` validation branches (score/confidence range errors)
- `tests/test_coverage_gaps.py`: 15 tests targeting previously uncovered branches (zero weights, empty n-grams, Jaccard edge cases)
- Coverage improved from 95% to **100%** across all 8 source modules

### Added — Cycle 2: Error Hardening

- `BaseDetector._validate_inputs()`: Type guard ensuring `claim` and `context` are always `str`; raises `TypeError` with descriptive message
- Input validation calls added to `OverlapDetector.detect()`, `EntailmentDetector.detect()`, `ConsistencyDetector.detect()`
- `GroundTruthDetector.detect_batch()`: Type validation for all claims/contexts elements
- `tests/test_error_hardening.py`: 18 tests covering None inputs, empty strings, huge strings (10k words), unicode edge cases, punctuation-only inputs

### Changed — Cycle 3: Performance

- `GroundTruthDetector.detect_batch()`: Parallelised via `ThreadPoolExecutor` — near-linear speedup for NLI-heavy workloads; order preserved via index mapping
- `GroundTruthDetector.__init__()`: Added `max_workers` parameter (configurable thread pool size)
- `OverlapDetector._tokenize()`: Decorated with `@lru_cache(maxsize=1024)` — avoids repeated regex evaluation on identical strings
- Added `cached_tokenize()` module-level utility in `aggregator.py` with `@lru_cache(maxsize=2048)`
- `tests/test_performance.py`: 9 timing tests confirming <100ms single detect, <2s for 50-pair batch, order preservation under parallelism

### Security — Cycle 4

- Scanned for hardcoded secrets: **0 found** (2 false positives filtered: word "token" in tokenize, lru_cache label)
- Scanned for injection vectors (subprocess, eval, exec, file I/O): **0 found**
- `tests/test_security.py`: 13 tests covering SQL injection strings, shell metacharacters, path traversal strings, null bytes, format string injection, integer/bytes/list type rejection, ReDoS safety, score integrity invariant

### Added — Cycle 5: CI/CD

- `.github/workflows/ci.yml`: GitHub Actions CI — checkout, setup-python 3.12, uv install, ruff lint, pytest with `--cov-fail-under=95`
- `.pre-commit-config.yaml`: ruff (format + check) and mypy pre-commit hooks

### Added — Cycle 6: Property-Based Testing

- `tests/test_property_based.py`: 15 Hypothesis property tests across 8 strategies:
  - Score/confidence always in [0, 1] for any string input (200+ examples per detector)
  - `is_hallucination` == `score > threshold` for any threshold in [0, 1]
  - No crashes on any random valid string (300+ examples per detector)
  - `DetectionResult` asdict() round-trip preserves all fields
  - Batch results always same length as inputs
  - Jaccard symmetry: `J(a,b) == J(b,a)` for all strings
  - Jaccard self-similarity: `J(a,a) == 1.0` for all strings
  - OverlapDetector threshold monotonicity
- **0 hypothesis-found failures** in underlying code

### Added — Cycle 7: Examples + Docs

- `examples/basic_detection.py`: Grounded vs. hallucinated claim comparison with per-detector breakdown
- `examples/batch_detection.py`: 8-pair batch detection with timing output and summary statistics
- `examples/custom_detector.py`: Custom NLI provider, weighted aggregation, ConsistencyDetector with sampled candidates, runtime `add_detector()`
- Complete docstrings added to `_ngram_recall()`, `OverlapDetector.detect()`, `_jaccard()`, `ConsistencyDetector.detect()`, `_validate_inputs()`

### Changed — Cycle 8: Release Engineering

- `pyproject.toml`: Added `authors`, `keywords`, PyPI `classifiers`; expanded `[dev]` extras to include `pytest-cov`, `hypothesis`, `mypy`
- `Makefile`: `test`, `lint`, `format`, `security`, `clean` targets
- `AGENTS.md`: Documents the Edgecraft autonomous development protocol
- `EVOLUTION.md`: Full cycle log with timestamps and findings
- `CHANGELOG.md`: This file

### Final State

| Metric | Value |
|--------|-------|
| Total tests | 112 |
| Test coverage | 100% |
| Property-based strategies | 8 (15 tests) |
| Security findings | 0 real (2 false positives) |
| Examples | 3 runnable scripts |
| CI pipeline | GitHub Actions on push + PR |
