"""
tests/test_tamper.py - Unit tests for pipeline/tamper.py.
Uses synthetic image pairs with known tamper regions. No API calls needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.schemas import BoundingBox


# -- Fixtures -----------------------------------------------------------------

def _canvas(h: int = 200, w: int = 300, val: int = 200) -> np.ndarray:
    """BGR canvas filled with a uniform grey value."""
    return np.ones((h, w, 3), dtype=np.uint8) * val


def _identity_flat() -> list[float]:
    return np.eye(3, dtype=np.float64).flatten().tolist()


# =============================================================================
#  build_text_mask
# =============================================================================

def test_mask_covers_box_interior():
    from pipeline.tamper import build_text_mask
    mask = build_text_mask(
        canvas_h=200, canvas_w=300,
        text_boxes=[BoundingBox(x=50, y=50, w=100, h=30)],
        pad=0,
    )
    assert mask[60, 80] == 255     # inside box
    assert mask[10, 10] == 0       # clearly outside

def test_mask_empty_boxes_all_zero():
    from pipeline.tamper import build_text_mask
    mask = build_text_mask(canvas_h=100, canvas_w=100, text_boxes=[])
    assert mask.sum() == 0

def test_mask_pad_extends_beyond_box():
    from pipeline.tamper import build_text_mask
    mask_no_pad  = build_text_mask(100, 200, [BoundingBox(x=50, y=50, w=40, h=20)], pad=0)
    mask_with_pad = build_text_mask(100, 200, [BoundingBox(x=50, y=50, w=40, h=20)], pad=10)
    # pad should increase the masked area
    assert mask_with_pad.sum() > mask_no_pad.sum()

def test_mask_clipped_to_canvas():
    """Box that extends beyond canvas edges should not raise."""
    from pipeline.tamper import build_text_mask
    mask = build_text_mask(
        canvas_h=100, canvas_w=100,
        text_boxes=[BoundingBox(x=90, y=90, w=50, h=50)],  # extends past 100x100
        pad=0,
    )
    assert mask.shape == (100, 100)

def test_mask_shape_matches_canvas():
    from pipeline.tamper import build_text_mask
    mask = build_text_mask(canvas_h=123, canvas_w=456, text_boxes=[])
    assert mask.shape == (123, 456)

def test_mask_multiple_boxes():
    from pipeline.tamper import build_text_mask
    boxes = [BoundingBox(x=10, y=10, w=20, h=10),
             BoundingBox(x=100, y=100, w=20, h=10)]
    mask  = build_text_mask(200, 300, boxes, pad=0)
    assert mask[15, 15]   == 255   # inside first box
    assert mask[105, 105] == 255   # inside second box
    assert mask[50, 50]   == 0     # between boxes


# =============================================================================
#  _cell_ssim
# =============================================================================

def test_ssim_identical_cells():
    from pipeline.tamper import _cell_ssim
    cell = np.random.randint(0, 255, (32, 32), dtype=np.uint8).astype(np.float32)
    assert _cell_ssim(cell, cell) == pytest.approx(1.0, abs=1e-4)

def test_ssim_completely_different():
    from pipeline.tamper import _cell_ssim
    a = np.zeros((32, 32), dtype=np.float32)
    b = np.ones ((32, 32), dtype=np.float32) * 255
    ssim = _cell_ssim(a, b)
    assert ssim < 0.5

def test_ssim_uniform_cells_return_one():
    """Two identical uniform cells should score 1.0 (zero variance case)."""
    from pipeline.tamper import _cell_ssim
    a = np.full((16, 16), 128.0, dtype=np.float32)
    assert _cell_ssim(a, a) == pytest.approx(1.0, abs=1e-4)


# =============================================================================
#  run_visual_diff
# =============================================================================

def test_identical_images_no_tamper():
    """Two identical images should produce zero tamper boxes."""
    from pipeline.tamper import build_text_mask, run_visual_diff
    img  = _canvas(200, 300, 180)
    mask = build_text_mask(200, 300, [])
    boxes = run_visual_diff(img, img, mask)
    assert boxes == []

def test_black_rectangle_detected():
    """A solid black rectangle added to the user image must be flagged."""
    from pipeline.tamper import build_text_mask, run_visual_diff
    ref  = _canvas(200, 300, 200)
    user = _canvas(200, 300, 200)
    user[120:170, 180:260] = 0     # black tamper rectangle

    mask  = build_text_mask(200, 300, [])
    boxes = run_visual_diff(ref, user, mask)
    assert len(boxes) > 0
    # At least one box should overlap the tampered region (y >= 100)
    assert any(b.y + b.h > 120 for b in boxes)

def test_text_masked_region_not_flagged():
    """Changes inside a fully text-masked region should not produce tamper boxes."""
    from pipeline.tamper import build_text_mask, run_visual_diff
    ref  = _canvas(200, 300, 200)
    user = _canvas(200, 300, 200)
    # Paint a bright rectangle in the user image
    user[10:50, 10:150] = 255
    # Mask that exact region as text
    mask  = build_text_mask(200, 300, [BoundingBox(x=10, y=10, w=140, h=40)], pad=0)
    boxes = run_visual_diff(ref, user, mask)
    assert boxes == []

def test_valid_mask_excludes_fill_pixels():
    """Fill pixels (outside warp boundary) must not produce tamper boxes."""
    from pipeline.tamper import build_text_mask, run_visual_diff
    ref  = _canvas(200, 300, 200)
    # user_aligned is all black (simulates complete warp fill)
    user_all_black = np.zeros((200, 300, 3), dtype=np.uint8)
    mask  = build_text_mask(200, 300, [])
    # Valid mask marks all pixels as invalid (fill)
    valid = np.zeros((200, 300), dtype=np.uint8)
    boxes = run_visual_diff(ref, user_all_black, mask, valid_mask=valid)
    assert boxes == []

def test_output_boxes_within_image_bounds():
    """All returned boxes must be within the canvas dimensions."""
    from pipeline.tamper import build_text_mask, run_visual_diff
    ref  = _canvas(200, 300, 200)
    user = _canvas(200, 300, 200)
    user[80:140, 80:200] = 0

    mask  = build_text_mask(200, 300, [])
    boxes = run_visual_diff(ref, user, mask)
    for b in boxes:
        assert b.x >= 0 and b.y >= 0
        assert b.x + b.w <= 300
        assert b.y + b.h <= 200


# =============================================================================
#  _scale_boxes
# =============================================================================

def test_scale_boxes_identity():
    from pipeline.tamper import _scale_boxes
    boxes = [BoundingBox(x=10, y=20, w=50, h=30)]
    assert _scale_boxes(boxes, 1.0) == boxes

def test_scale_boxes_half():
    from pipeline.tamper import _scale_boxes
    boxes  = [BoundingBox(x=100, y=200, w=50, h=30)]
    result = _scale_boxes(boxes, 0.5)
    assert result[0].x == 200   # x / scale
    assert result[0].y == 400
    assert result[0].w == 100

def test_scale_boxes_min_dimension_one():
    from pipeline.tamper import _scale_boxes
    boxes  = [BoundingBox(x=0, y=0, w=1, h=1)]
    result = _scale_boxes(boxes, 0.5)
    assert result[0].w >= 1
    assert result[0].h >= 1


# =============================================================================
#  _project_boxes_to_user
# =============================================================================

def test_project_boxes_identity_homography():
    from pipeline.tamper import _project_boxes_to_user
    H_inv = np.eye(3, dtype=np.float64)
    boxes = [BoundingBox(x=50, y=50, w=100, h=80)]
    result = _project_boxes_to_user(boxes, H_inv, user_h=500, user_w=600)
    assert result[0].x == pytest.approx(50, abs=2)
    assert result[0].y == pytest.approx(50, abs=2)

def test_project_boxes_clipped():
    from pipeline.tamper import _project_boxes_to_user
    H_inv = np.eye(3, dtype=np.float64)
    boxes = [BoundingBox(x=550, y=450, w=200, h=200)]
    result = _project_boxes_to_user(boxes, H_inv, user_h=500, user_w=600)
    assert result[0].x + result[0].w <= 600
    assert result[0].y + result[0].h <= 500


# =============================================================================
#  run() integration
# =============================================================================

class _FakeSemanticMap:
    """Minimal SemanticMap stand-in for testing run()."""
    def __init__(self, words=None):
        class _W:
            def __init__(self, bbox): self.bbox = bbox
        self.words = [_W(b) for b in (words or [])]

def test_run_identical_images_no_boxes():
    """Identical images with identity homography -> no tamper boxes."""
    from pipeline.tamper import run
    img = _canvas(300, 400, 180)
    H   = _identity_flat()
    ref_map = _FakeSemanticMap()
    boxes_ref, boxes_user = run(img, img, H, H, ref_map, inlier_ratio=0.9)
    assert boxes_ref  == []
    assert boxes_user == []

def test_run_returns_tuple_of_lists():
    from pipeline.tamper import run
    img = _canvas(200, 300, 200)
    H   = _identity_flat()
    ref_map = _FakeSemanticMap()
    result = run(img, img, H, H, ref_map, inlier_ratio=0.9)
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)

def test_run_no_homography_still_works():
    """When H is None, run() should complete without raising."""
    from pipeline.tamper import run
    img     = _canvas(200, 300, 200)
    ref_map = _FakeSemanticMap()
    boxes_ref, boxes_user = run(img, img, None, None, ref_map, inlier_ratio=0.0)
    assert isinstance(boxes_ref,  list)
    assert isinstance(boxes_user, list)
    assert boxes_user == []   # no H_inv -> no user boxes

def test_run_tamper_detected_on_modified_image():
    """Adding a black rectangle in the user image should produce tamper boxes."""
    from pipeline.tamper import run
    ref  = _canvas(300, 400, 180)
    user = _canvas(300, 400, 180)
    user[150:220, 200:320] = 0   # large black tamper patch

    H       = _identity_flat()
    ref_map = _FakeSemanticMap()
    boxes_ref, _ = run(ref, user, H, H, ref_map, inlier_ratio=0.9)
    assert len(boxes_ref) > 0

def test_run_text_mask_suppresses_text_region_tamper():
    """Changes inside word bbox regions should not produce tamper boxes."""
    from pipeline.tamper import run
    ref  = _canvas(300, 400, 180)
    user = _canvas(300, 400, 180)
    # Paint a bright patch in a region covered by a word box
    user[50:90, 50:200] = 255
    word_box = BoundingBox(x=50, y=50, w=150, h=40)
    ref_map  = _FakeSemanticMap(words=[word_box])

    H = _identity_flat()
    boxes_ref, _ = run(ref, user, H, H, ref_map, inlier_ratio=0.9)
    assert boxes_ref == []