"""
tests/test_diff.py — Unit tests for pipeline/diff.py.

These tests run against pure Python logic (no images, no OCR, no API calls).
Each test builds minimal WordEntry fixtures and asserts DiffEntry outputs.
"""
from __future__ import annotations

import pytest

from models.schemas import BoundingBox, DiffType, Polygon, WordEntry


def _word(text: str, x: int = 0, y: int = 0, w: int = 50, h: int = 20) -> WordEntry:
    """Helper — build a minimal WordEntry for testing."""
    return WordEntry(
        text=text,
        confidence=0.99,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        polygon=Polygon(points=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)]),
    )


# ── levenshtein ───────────────────────────────────────────────────────────────

def test_levenshtein_identical():
    from pipeline.diff import levenshtein
    assert levenshtein("hello", "hello") == 0

def test_levenshtein_empty():
    from pipeline.diff import levenshtein
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3

def test_levenshtein_one_edit():
    from pipeline.diff import levenshtein
    assert levenshtein("cat", "bat") == 1


# ── similarity ────────────────────────────────────────────────────────────────

def test_similarity_identical():
    from pipeline.diff import similarity
    assert similarity("Amber", "Amber") == pytest.approx(1.0)

def test_similarity_empty_both():
    from pipeline.diff import similarity
    assert similarity("", "") == pytest.approx(1.0)

def test_similarity_case_insensitive():
    from pipeline.diff import similarity
    assert similarity("AMBER", "amber") == pytest.approx(1.0)


# ── fuzzy_lcs_diff ────────────────────────────────────────────────────────────

def test_all_match():
    """Identical word lists should produce all 'match' entries."""
    from pipeline.diff import fuzzy_lcs_diff
    wa = [_word("hello"), _word("world")]
    wb = [_word("hello"), _word("world")]
    result = fuzzy_lcs_diff(wa, wb)
    assert all(d.type == DiffType.match for d in result)

def test_single_removal():
    from pipeline.diff import fuzzy_lcs_diff
    wa = [_word("hello"), _word("world")]
    wb = [_word("hello")]
    result = fuzzy_lcs_diff(wa, wb)
    types = [d.type for d in result]
    assert DiffType.removed in types
    assert DiffType.added not in types

def test_single_addition():
    from pipeline.diff import fuzzy_lcs_diff
    wa = [_word("hello")]
    wb = [_word("hello"), _word("world")]
    result = fuzzy_lcs_diff(wa, wb)
    types = [d.type for d in result]
    assert DiffType.added in types

def test_modification():
    """'Maber' vs 'Amber' should be a modification, not remove+add."""
    from pipeline.diff import fuzzy_lcs_diff
    wa = [_word("Amber")]
    wb = [_word("Maber")]
    result = fuzzy_lcs_diff(wa, wb)
    assert len(result) == 1
    assert result[0].type == DiffType.modified


# ── spatial_second_pass ───────────────────────────────────────────────────────

def test_spatial_pass_promotes_colocated():
    """Removed/added words at the same position should become 'modified'."""
    from pipeline.diff import fuzzy_lcs_diff, spatial_second_pass
    wa = [_word("OldText", x=100, y=100)]
    wb = [_word("NewText", x=105, y=102)]
    # LCS will mark these as removed + added (different enough text)
    diff = fuzzy_lcs_diff(wa, wb)
    result = spatial_second_pass(diff, ref_h=500, ref_w=500, user_h=500, user_w=500)
    types = [d.type for d in result]
    assert DiffType.modified in types

def test_spatial_pass_no_cross_column():
    """Words in different columns must NOT be paired by spatial pass."""
    from pipeline.diff import fuzzy_lcs_diff, spatial_second_pass
    wa = [_word("Left",  x=50,  y=100)]
    wb = [_word("Right", x=450, y=100)]
    diff   = fuzzy_lcs_diff(wa, wb)
    result = spatial_second_pass(diff, ref_h=500, ref_w=500, user_h=500, user_w=500)
    types  = [d.type for d in result]
    # Should remain removed + added, not be promoted to modified
    assert DiffType.modified not in types


# ── suppress_long_runs ────────────────────────────────────────────────────────

def test_long_run_collapsed():
    from pipeline.diff import fuzzy_lcs_diff, suppress_long_runs
    # 6 consecutive additions should collapse to unmatched_run
    wa = [_word("anchor")]
    wb = [_word("anchor")] + [_word(f"word{i}") for i in range(6)]
    diff   = fuzzy_lcs_diff(wa, wb)
    result = suppress_long_runs(diff)
    types  = [d.type for d in result]
    assert DiffType.unmatched_run in types