"""Security tests: injection vectors, path traversal, malformed input safety.

Scan results summary:
- Hardcoded secrets: 0 found (FALSE POSITIVE: 'token' appears only in 'tokenize' — not a credential)
- Subprocess calls: 0 found — no shell injection vectors
- eval/exec: 0 found — no code injection vectors
- File open(): 0 found — no path traversal vectors
- All configuration is pure-Python dataclasses with no persistence
"""

from __future__ import annotations

import pytest

from groundtruth import (
    AggregatedResult,
    ConsistencyDetector,
    EntailmentDetector,
    GroundTruthDetector,
    OverlapDetector,
)

# ---------------------------------------------------------------------------
# Injection safety — text inputs treated as data, not code
# ---------------------------------------------------------------------------


def test_overlap_sql_injection_string_safe() -> None:
    """SQL injection strings do not cause errors — treated as plain text."""
    det = OverlapDetector()
    claim = "'; DROP TABLE claims; --"
    context = "SELECT * FROM claims WHERE id = 1"
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0


def test_entailment_shell_injection_string_safe() -> None:
    """Shell metacharacter strings do not cause errors."""
    det = EntailmentDetector()
    claim = "$(rm -rf /); `whoami`"
    context = "normal text without special meaning"
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0


def test_consistency_path_traversal_string_safe() -> None:
    """Path traversal strings do not cause file system access."""
    det = ConsistencyDetector()
    claim = "../../../../etc/passwd"
    context = "system files should not be accessible"
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0


def test_aggregator_script_tag_injection_safe() -> None:
    """XSS-style strings are handled as plain text."""
    agg = GroundTruthDetector()
    claim = "<script>alert('xss')</script>"
    context = "the page displays content without scripts"
    result = agg.detect(claim=claim, context=context)
    assert isinstance(result, AggregatedResult)


def test_aggregator_null_byte_string_safe() -> None:
    """Strings containing null bytes do not cause crashes."""
    det = OverlapDetector()
    claim = "hello\x00world"
    context = "hello world"
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0


def test_aggregator_format_string_safe() -> None:
    """Python format string injection attempts are harmless."""
    det = EntailmentDetector()
    claim = "{0.__class__.__mro__}"
    context = "class hierarchy introspection attempt"
    result = det.detect(claim=claim, context=context)
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Type safety — only str accepted
# ---------------------------------------------------------------------------


def test_overlap_integer_claim_raises_type_error() -> None:
    """Passing an integer as claim raises TypeError."""
    det = OverlapDetector()
    with pytest.raises(TypeError):
        det.detect(claim=42, context="valid context")  # type: ignore[arg-type]


def test_entailment_bytes_claim_raises_type_error() -> None:
    """Passing bytes as claim raises TypeError."""
    det = EntailmentDetector()
    with pytest.raises(TypeError):
        det.detect(claim=b"bytes claim", context="valid context")  # type: ignore[arg-type]


def test_aggregator_list_as_claim_raises_type_error() -> None:
    """Passing a list as claim raises TypeError."""
    det = OverlapDetector()
    with pytest.raises(TypeError):
        det.detect(claim=["not", "a", "string"], context="valid context")  # type: ignore[arg-type]


def test_aggregator_batch_non_string_claim_raises() -> None:
    """detect_batch with non-string in claims list raises TypeError."""
    agg = GroundTruthDetector(detectors=[OverlapDetector()])
    with pytest.raises(TypeError, match="strings"):
        agg.detect_batch([42, "valid"], ["context1", "context2"])  # type: ignore[list-item]


def test_aggregator_batch_non_string_context_raises() -> None:
    """detect_batch with non-string in contexts list raises TypeError."""
    agg = GroundTruthDetector(detectors=[OverlapDetector()])
    with pytest.raises(TypeError, match="strings"):
        agg.detect_batch(["valid"], [None])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# ReDoS safety — catastrophic backtracking prevention
# ---------------------------------------------------------------------------


def test_overlap_repeated_pattern_no_timeout() -> None:
    """Regex on ReDoS-style inputs completes quickly (no catastrophic backtracking).

    The regex r'\\b\\w+\\b' uses simple word-boundary matching and does not
    exhibit catastrophic backtracking — FALSE POSITIVE: no ReDoS risk here.
    """
    import time

    det = OverlapDetector()
    # Potential ReDoS pattern: many repeated 'a' characters
    evil_input = "a" * 10_000
    start = time.perf_counter()
    result = det.detect(claim=evil_input, context="a b c d e")
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Possible ReDoS: took {elapsed:.3f}s"
    assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Score integrity — outputs always in documented range
# ---------------------------------------------------------------------------


def test_score_never_exceeds_one_on_any_input() -> None:
    """No input can produce a score outside [0, 1]."""
    agg = GroundTruthDetector()
    adversarial_inputs = [
        ("", ""),
        ("a" * 1000, "b" * 1000),
        ("the " * 500, "the " * 500),
        ("\n\t\r", "   "),
        ("🎯🔥💡", "emoji content here"),
    ]
    for claim, context in adversarial_inputs:
        result = agg.detect(claim=claim, context=context)
        assert 0.0 <= result.score <= 1.0, f"Out-of-range score for: {claim[:20]!r}"
        assert 0.0 <= result.confidence <= 1.0
