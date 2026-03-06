"""
tests/test_align.py — Unit tests for pipeline/align.py.
Uses synthetic images to verify homography and coordinate projection.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.schemas import BoundingBox


def _make_checkerboard(h: int = 200, w: int = 300) -> np.ndarray:
    """Generate a checkerboard pattern for ORB to find keypoints in."""
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cell = 20
    for r in range(0, h, cell):
        for c in range(0, w, cell):
            if (r // cell + c // cell) % 2 == 0:
                img[r:r+cell, c:c+cell] = 200
    return img


def test_identity_homography():
    """Identical images should produce a near-identity homography."""
    from pipeline.align import compute_homography
    img    = _make_checkerboard()
    result = compute_homography(img, img)
    assert result.status == "ok"
    assert result.inlier_ratio > 0.8


def test_project_boxes_identity():
    """project_boxes with identity H_inv should return same coordinates."""
    from pipeline.align import project_boxes
    identity = np.eye(3).flatten().tolist()
    boxes  = [BoundingBox(x=10, y=20, w=50, h=30)]
    result = project_boxes(boxes, identity, clip_w=500, clip_h=500)
    assert result[0].x == pytest.approx(10, abs=2)
    assert result[0].y == pytest.approx(20, abs=2)


def test_project_boxes_clipping():
    """Boxes projecting outside image bounds should be clipped."""
    from pipeline.align import project_boxes
    identity = np.eye(3).flatten().tolist()
    boxes  = [BoundingBox(x=490, y=490, w=100, h=100)]
    result = project_boxes(boxes, identity, clip_w=500, clip_h=500)
    assert result[0].x + result[0].w <= 500
    assert result[0].y + result[0].h <= 500