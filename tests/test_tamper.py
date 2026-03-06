"""
tests/test_tamper.py — Unit tests for pipeline/tamper.py.
Uses synthetic image pairs with known tamper regions.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.schemas import BoundingBox


def _white_canvas(h: int = 200, w: int = 300) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * 240


def test_build_text_mask_covers_boxes():
    """Text mask should be 255 within padded box regions."""
    from pipeline.tamper import build_text_mask
    mask = build_text_mask(
        canvas_h=200, canvas_w=300,
        text_boxes=[BoundingBox(x=50, y=50, w=100, h=30)],
        pad=0,
    )
    assert mask[60, 80] == 255     # inside box
    assert mask[10, 10] == 0       # outside box


def test_build_text_mask_empty():
    """Empty box list should produce all-zero mask."""
    from pipeline.tamper import build_text_mask
    mask = build_text_mask(canvas_h=100, canvas_w=100, text_boxes=[])
    assert mask.sum() == 0


def test_black_rectangle_detected():
    """A black rectangle added to an otherwise identical image should be flagged."""
    from pipeline.tamper import build_text_mask, run_visual_diff
    ref  = _white_canvas()
    user = _white_canvas()
    # Add a black rectangle to the user image (tamper simulation)
    user[120:160, 200:260] = 0

    mask  = build_text_mask(100, 100, [])   # no text to mask
    boxes = run_visual_diff(ref, user, np.zeros((200, 300), dtype=np.uint8))
    # At least one box should overlap the tampered region
    tamper_ys = [b.y for b in boxes]
    assert any(y >= 100 for y in tamper_ys)