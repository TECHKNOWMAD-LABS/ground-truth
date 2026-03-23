# AGENTS.md — Edgecraft Autonomous Development Protocol

This file documents the autonomous development protocol used to evolve this
codebase from its initial scaffold to a production-grade release.

## Protocol Overview

**Edgecraft** is an 8-cycle autonomous iteration system developed by
TechKnowMad Labs. Each cycle addresses a distinct quality dimension of the
codebase. Cycles are executed sequentially by an AI agent with no human
intervention between steps.

## Layer Prefix Convention

All commits produced by Edgecraft use a standardised layer prefix that maps
to a cognitive layer in the system's reasoning chain:

| Prefix | Layer | Purpose |
|--------|-------|---------|
| `L0/attention:` | Attention | Initial focus / context loading |
| `L1/detection:` | Detection | Identifying problems, gaps, or risks |
| `L2/noise:` | Noise | Filtering false positives from signal |
| `L3/sub-noise:` | Sub-noise | Subtle bugs or edge cases |
| `L4/conjecture:` | Conjecture | Hypotheses before implementation |
| `L5/action:` | Action | Implementation of fixes or features |
| `L6/grounding:` | Grounding | Verification via tests and measurements |
| `L7/flywheel:` | Flywheel | Patterns that generalise to other modules |

## The 8 Cycles

### Cycle 1 — Test Coverage
Identify all source files with zero or partial coverage. Create `conftest.py`
with shared fixtures. Write tests for every uncovered branch. Target: 100%
coverage.

### Cycle 2 — Error Hardening
Attempt to break the code with adversarial inputs (None, empty, huge strings,
unicode, punctuation-only). Add `TypeError` guards, input validation in base
class, and type checks in batch operations.

### Cycle 3 — Performance
Profile sequential bottlenecks. Parallelise batch operations via
`ThreadPoolExecutor`. Add `lru_cache` on hot tokenization paths. Validate
with timing assertions.

### Cycle 4 — Security
Scan for hardcoded secrets (28+ patterns), injection vectors (subprocess,
eval, exec), and path traversal risks. Document true findings and false
positives. Add type-safety and score-integrity tests.

### Cycle 5 — CI/CD
Create GitHub Actions workflow (`ci.yml`) with lint + test gates. Add
`.pre-commit-config.yaml` with ruff and mypy hooks.

### Cycle 6 — Property-Based Testing
Use Hypothesis to verify core invariants across thousands of generated inputs:
score bounds, symmetry, monotonicity, round-trips, no-crash guarantees.

### Cycle 7 — Examples + Docs
Create 2-3 runnable example scripts covering the full public API. Add
complete docstrings to every public function. Verify all examples execute
without errors.

### Cycle 8 — Release Engineering
Ensure `pyproject.toml` metadata is complete. Create `CHANGELOG.md`,
`Makefile`, `AGENTS.md`, and `EVOLUTION.md`. Tag `v0.1.0`.

## Execution Model

Each step in each cycle follows this sequence:

1. **Read** — explore relevant source files
2. **Analyse** — identify the gap or improvement
3. **Write** — implement the change
4. **Test** — run pytest; fix failures before committing
5. **Commit** — meaningful diff only, with Edgecraft layer prefix
6. **Push** — after each cycle to ensure remote is always current

## Agent Identity

- System: Edgecraft Protocol v1.0
- Operator: TechKnowMad Labs (`admin@techknowmad.ai`)
- Model: Claude Sonnet (claude-sonnet-4-6)
- Execution date: 2026-03-23
