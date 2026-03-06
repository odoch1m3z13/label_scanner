"""
pipeline/words.py — Token normalisation and reading order utilities.

Operates on WordEntry lists produced by workers/cloud_vision.py.
All functions are pure (no I/O, no side effects).
"""
from __future__ import annotations

import re

import numpy as np

from models.schemas import BoundingBox, Polygon, WordEntry


def clean(text: str) -> str:
    """Lowercase alphanumeric only — used for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sort_reading_order(words: list[WordEntry]) -> list[WordEntry]:
    """
    Sort words into natural reading order (top-to-bottom, left-to-right).
    Uses median word height as the row-snap tolerance so words on the same
    visual line are grouped together before sorting by x.
    """
    if not words:
        return words

    # TODO: implement
    # median_h = max(10.0, float(np.median([w.bbox.h for w in words])))
    # return sorted(words, key=lambda w: (
    #     int(w.bbox.y / median_h),
    #     w.bbox.x,
    # ))
    raise NotImplementedError


def filter_low_confidence(
    words:      list[WordEntry],
    min_conf:   float,
) -> list[WordEntry]:
    """Remove words below min_conf threshold."""
    return [w for w in words if w.confidence >= min_conf]


def merge_region_words(
    primary:   list[WordEntry],
    secondary: list[WordEntry],
    region:    BoundingBox,
) -> list[WordEntry]:
    """
    Replace words whose centres fall inside `region` with the secondary list,
    then re-sort in reading order.

    Used by the nutrition-panel second-pass merge (if re-added), or any
    scenario where a targeted sub-region extraction overrides the global pass.
    """
    # TODO: implement
    # rx1, ry1 = region.x, region.y
    # rx2, ry2 = rx1 + region.w, ry1 + region.h
    # outside = [
    #     w for w in primary
    #     if not (rx1 <= w.bbox.x + w.bbox.w / 2 <= rx2 and
    #             ry1 <= w.bbox.y + w.bbox.h / 2 <= ry2)
    # ]
    # return sort_reading_order(outside + secondary)
    raise NotImplementedError