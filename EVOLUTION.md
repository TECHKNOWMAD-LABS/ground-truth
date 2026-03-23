# EVOLUTION.md — Edgecraft Cycle Log

Full log of all 8 autonomous iteration cycles for `ground-truth`.

**Execution date:** 2026-03-23
**Operator:** TechKnowMad Labs (admin@techknowmad.ai)
**Protocol:** Edgecraft v1.0

---

## Cycle 1 — Test Coverage

**Timestamp:** 2026-03-23T00:01
**Duration:** ~5 minutes

**Findings:**
- 4 modules at <100% coverage: `models.py` (88%), `aggregator.py` (95%), `consistency.py` (92%), `overlap.py` (97%)
- Missing: `DetectionResult` and `AggregatedResult` validation error paths
- Missing: Aggregator zero-weights branch, negative add_detector weight
- Missing: Jaccard both-empty and one-empty edge cases
- Missing: `_ngram_recall` with empty claim n-grams

**Actions:**
- Created `tests/conftest.py` with shared `FixedNLI`, `ClampingNLI`, detector fixtures
- Created `tests/test_models.py` — 10 tests
- Created `tests/test_coverage_gaps.py` — 15 tests

**Result:** Coverage 95% → **100%** | Tests: 31 → **57**

---

## Cycle 2 — Error Hardening

**Timestamp:** 2026-03-23T00:06
**Duration:** ~5 minutes

**Findings:**
- No `TypeError` guard on `detect()` — `None` inputs produce confusing `AttributeError`
- No validation for `detect_batch()` element types

**Actions:**
- Added `BaseDetector._validate_inputs()` static method with clear `TypeError` messages
- Added validation calls in all 3 detector `detect()` methods
- Added element-type checks in `GroundTruthDetector.detect_batch()`
- Created `tests/test_error_hardening.py` — 18 tests (None, empty, huge, unicode, punctuation)

**Result:** Tests: 57 → **75** | 0 failures

---

## Cycle 3 — Performance

**Timestamp:** 2026-03-23T00:11
**Duration:** ~7 minutes

**Conjecture:** Parallelizing N `detect()` calls in `detect_batch()` will yield ~Nx speedup for NLI-heavy workloads.

**Actions:**
- Refactored `detect_batch()` to use `ThreadPoolExecutor` with index-based result ordering
- Added `max_workers` parameter to `GroundTruthDetector.__init__()`
- Added `@lru_cache(maxsize=1024)` to `OverlapDetector._tokenize()`
- Added `cached_tokenize()` module-level utility with `@lru_cache(maxsize=2048)`
- Created `tests/test_performance.py` — 9 timing and correctness tests

**Measured results:**
- Single detect: <1ms for short texts (<100ms threshold)
- 50-pair batch: <10ms (well under 2s threshold)
- Parallel correctness: identical results to sequential

**Flywheel:** `lru_cache` pattern applicable to `_jaccard` and `_key_terms` functions.

---

## Cycle 4 — Security

**Timestamp:** 2026-03-23T00:18
**Duration:** ~5 minutes

**Scan results:**
- Hardcoded secrets: **0 real findings**
- False positives filtered: 2 (word "token" in tokenize function name; lru_cache label)
- Subprocess/eval/exec: **0 found**
- File open() calls: **0 found**
- Path traversal: **0 risk** (no file I/O)

**Actions:**
- Created `tests/test_security.py` — 13 tests:
  - SQL injection, shell metacharacters, path traversal, null bytes, format strings → all safe
  - Integer, bytes, list as claim/context → `TypeError`
  - ReDoS safety: `\b\w+\b` does not backtrack catastrophically
  - Score integrity: `[0,1]` on all adversarial inputs

**Result:** Tests: 75 → **97** | 0 security vulnerabilities

---

## Cycle 5 — CI/CD

**Timestamp:** 2026-03-23T00:23
**Duration:** ~3 minutes

**Actions:**
- Created `.github/workflows/ci.yml`: checkout → setup-python 3.12 → uv install → ruff check → pytest with `--cov-fail-under=95`
- Created `.pre-commit-config.yaml`: ruff (format + check) + mypy hooks
- Fixed 13 ruff lint issues in test files (unused imports, import ordering)

**Result:** CI pipeline active on every push and PR

---

## Cycle 6 — Property-Based Testing

**Timestamp:** 2026-03-23T00:26
**Duration:** ~5 minutes

**Actions:**
- Created `tests/test_property_based.py` — 15 Hypothesis tests across 8 strategies:
  1. Score in `[0,1]` for all detectors — 200+ examples each
  2. `is_hallucination == score > threshold` for any threshold
  3. No crashes on any random valid string — 300+ examples
  4. `DetectionResult` asdict() round-trip fidelity
  5. Batch length always == input length
  6. Jaccard symmetry `J(a,b) == J(b,a)`
  7. Jaccard self-similarity `J(a,a) == 1.0`
  8. OverlapDetector threshold monotonicity

**Hypothesis findings:** 0 failures — all invariants hold

**Result:** Tests: 97 → **112** | 15 property tests green

---

## Cycle 7 — Examples + Docs

**Timestamp:** 2026-03-23T00:31
**Duration:** ~5 minutes

**Actions:**
- Created `examples/basic_detection.py`: basic grounded vs. hallucinated detection
- Created `examples/batch_detection.py`: 8-pair batch with timing output
- Created `examples/custom_detector.py`: custom NLI, weighted aggregation, runtime add_detector
- All 3 examples tested — execute without errors
- Added full docstrings to: `_ngram_recall`, `OverlapDetector.detect`, `_jaccard`, `ConsistencyDetector.detect`, `BaseDetector._validate_inputs`

**Result:** Complete API documentation; 3 runnable examples

---

## Cycle 8 — Release Engineering

**Timestamp:** 2026-03-23T00:36
**Duration:** ~5 minutes

**Actions:**
- `pyproject.toml`: added `authors`, `keywords`, PyPI classifiers; expanded dev deps
- Created `CHANGELOG.md`: full change history with metrics table
- Created `Makefile`: `test`, `test-fast`, `test-property`, `lint`, `format`, `security`, `clean`, `coverage-html`, `examples`
- Created `AGENTS.md`: protocol documentation
- Created `EVOLUTION.md`: this file
- Tagged `v0.1.0`

**Final metrics:**

| Metric | Value |
|--------|-------|
| Total commits | 16 |
| Total tests | 112 |
| Test coverage | 100% |
| Property-based tests | 15 |
| Security findings | 0 |
| Examples | 3 |
| Source files modified | 6 |
| New files created | 14 |
