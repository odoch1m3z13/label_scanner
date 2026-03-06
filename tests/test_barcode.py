"""
tests/test_barcode.py — HOG cosine similarity threshold validation.
Uses synthetic bar patterns to verify the threshold separates same vs different.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_barcode(w: int = 128, h: int = 64, bar_w: int = 4) -> np.ndarray:
    """Generate a synthetic barcode-like vertical bar pattern."""
    img = np.zeros((h, w), dtype=np.uint8)
    for x in range(0, w, bar_w * 2):
        img[:, x:x+bar_w] = 255
    return img


def test_same_barcode_high_similarity():
    """Identical bar patterns should score above BARCODE_HOG_THRESH."""
    from pipeline.barcode import cosine_sim, hog_vec
    from config import BARCODE_HOG_THRESH
    bc = _make_barcode()
    sim = cosine_sim(hog_vec(bc), hog_vec(bc))
    assert sim >= BARCODE_HOG_THRESH


def test_different_barcode_low_similarity():
    """Different bar spacings should score below BARCODE_HOG_THRESH."""
    from pipeline.barcode import cosine_sim, hog_vec
    from config import BARCODE_HOG_THRESH
    bc1 = _make_barcode(bar_w=4)
    bc2 = _make_barcode(bar_w=12)   # very different spacing
    sim = cosine_sim(hog_vec(bc1), hog_vec(bc2))
    assert sim < BARCODE_HOG_THRESH


def test_solid_black_low_similarity():
    """A black rectangle replacing a barcode should score very low."""
    from pipeline.barcode import cosine_sim, hog_vec
    from config import BARCODE_HOG_THRESH
    bc    = _make_barcode()
    black = np.zeros((64, 128), dtype=np.uint8)
    sim   = cosine_sim(hog_vec(bc), hog_vec(black))
    assert sim < BARCODE_HOG_THRESH


def test_hog_vec_empty_crop():
    """Empty crop should return a zero vector without raising."""
    from pipeline.barcode import hog_vec
    vec = hog_vec(np.zeros((0, 0), dtype=np.uint8))
    assert vec is not None