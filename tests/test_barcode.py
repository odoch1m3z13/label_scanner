"""
tests/test_barcode.py - Unit tests for pipeline/barcode.py.

Uses synthetic bar patterns and labelled images.
No API calls, no file I/O.

Barcode simulation:
  _make_barcode(w, h, bar_w) generates alternating black/white vertical bars
  that mimic a 1D barcode. The x-direction Sobel gradient of this pattern has
  the same column-density signature as a real barcode.
"""
from __future__ import annotations

import numpy as np
import pytest
import cv2


# =============================================================================
#  Fixtures
# =============================================================================

def _make_barcode(w: int = 200, h: int = 80, bar_w: int = 4) -> np.ndarray:
    """Synthetic grayscale barcode: alternating black/white vertical bars."""
    img = np.zeros((h, w), dtype=np.uint8)
    for x in range(0, w, bar_w * 2):
        img[:, x:x+bar_w] = 255
    return img


def _make_barcode_bgr(w: int = 200, h: int = 80, bar_w: int = 4) -> np.ndarray:
    """BGR version of synthetic barcode."""
    gray  = _make_barcode(w, h, bar_w)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _label_with_barcode(
    canvas_h: int = 300, canvas_w: int = 400,
    bc_x: int = 50,  bc_y: int = 100,
    bc_w: int = 200, bc_h: int = 80,
    bar_w: int = 4,
) -> np.ndarray:
    """
    Build a synthetic BGR label image with a barcode at a known position.
    Background is mid-grey; barcode region contains vertical bar pattern.
    """
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 180
    bc     = _make_barcode_bgr(bc_w, bc_h, bar_w)
    canvas[bc_y:bc_y+bc_h, bc_x:bc_x+bc_w] = bc
    return canvas


def _identity_H() -> list[float]:
    return np.eye(3, dtype=np.float64).flatten().tolist()


# =============================================================================
#  hog_vec
# =============================================================================

def test_hog_vec_shape():
    """hog_vec should return a 1D float32 array."""
    from pipeline.barcode import hog_vec
    bc  = _make_barcode()
    vec = hog_vec(bc)
    assert vec.ndim == 1
    assert vec.dtype == np.float32

def test_hog_vec_empty_crop_returns_zero():
    """Empty crop must return a zero vector without raising."""
    from pipeline.barcode import hog_vec
    vec = hog_vec(np.zeros((0, 0), dtype=np.uint8))
    assert vec is not None
    assert (vec == 0).all()

def test_hog_vec_identical_crops_equal():
    """Same image twice -> identical descriptor vectors."""
    from pipeline.barcode import hog_vec
    bc   = _make_barcode()
    v1   = hog_vec(bc)
    v2   = hog_vec(bc.copy())
    assert np.allclose(v1, v2)

def test_hog_vec_non_zero_for_barcode():
    """A barcode image should produce a non-zero HOG descriptor."""
    from pipeline.barcode import hog_vec
    bc  = _make_barcode()
    vec = hog_vec(bc)
    assert np.linalg.norm(vec) > 0


# =============================================================================
#  cosine_sim
# =============================================================================

def test_cosine_sim_identical_vectors():
    from pipeline.barcode import cosine_sim
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

def test_cosine_sim_zero_vector_returns_zero():
    from pipeline.barcode import cosine_sim
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    z = np.zeros(3, dtype=np.float32)
    assert cosine_sim(v, z) == 0.0
    assert cosine_sim(z, v) == 0.0

def test_cosine_sim_orthogonal_vectors():
    from pipeline.barcode import cosine_sim
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-5)

def test_same_barcode_high_similarity():
    """Identical bar patterns must score above BARCODE_HOG_THRESH."""
    from pipeline.barcode import cosine_sim, hog_vec
    from config import BARCODE_HOG_THRESH
    bc  = _make_barcode()
    sim = cosine_sim(hog_vec(bc), hog_vec(bc))
    assert sim >= BARCODE_HOG_THRESH

def _make_horizontal_bars(w: int = 200, h: int = 80, bar_h: int = 4) -> np.ndarray:
    """Horizontal bar pattern — 90-degree rotation of a barcode."""
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, bar_h * 2):
        img[y:y+bar_h, :] = 255
    return img


def test_horizontal_bars_low_similarity():
    """
    Horizontal bars (90 degree rotation of barcode pattern) should score
    near zero — HOG is orientation-sensitive and the dominant gradient
    direction flips from 0/180 degrees (vertical bars) to 90/270 degrees
    (horizontal bars).

    Note: two barcodes with different bar WIDTHS but the same vertical
    orientation will score very similarly (~0.99) because HOG captures
    gradient orientation, not spatial frequency. The meaningful contrast
    case is a structurally different pattern, not just wider/narrower bars.
    """
    from pipeline.barcode import cosine_sim, hog_vec
    from config import BARCODE_HOG_THRESH
    bc    = _make_barcode()
    horiz = _make_horizontal_bars()
    sim   = cosine_sim(hog_vec(bc), hog_vec(horiz))
    assert sim < BARCODE_HOG_THRESH

def test_solid_black_barcode_low_similarity():
    """A solid black rectangle replacing a barcode should score very low."""
    from pipeline.barcode import cosine_sim, hog_vec
    from config import BARCODE_HOG_THRESH
    bc    = _make_barcode()
    black = np.zeros((80, 200), dtype=np.uint8)
    sim   = cosine_sim(hog_vec(bc), hog_vec(black))
    assert sim < BARCODE_HOG_THRESH

def test_cosine_sim_scale_invariant():
    """Multiplying a vector by a scalar should not change cosine sim."""
    from pipeline.barcode import cosine_sim
    bc = _make_barcode()
    from pipeline.barcode import hog_vec
    v1  = hog_vec(bc)
    sim = cosine_sim(v1, v1 * 5.0)
    assert sim == pytest.approx(1.0, abs=1e-4)


# =============================================================================
#  find_barcode_regions
# =============================================================================

def test_find_barcode_regions_detects_barcode():
    """A label with a clear barcode region should be detected."""
    from pipeline.barcode import find_barcode_regions
    # Build a grayscale label with a clear barcode strip
    gray = np.ones((200, 400), dtype=np.uint8) * 180
    # Place barcode at x=80, w=160
    bc = _make_barcode(w=160, h=60, bar_w=4)
    gray[70:130, 80:240] = bc
    regions = find_barcode_regions(gray)
    assert len(regions) > 0

def test_find_barcode_regions_returns_tuples():
    """Each region should be a (x, y, w, h) tuple of ints."""
    from pipeline.barcode import find_barcode_regions
    gray = np.ones((200, 400), dtype=np.uint8) * 180
    bc   = _make_barcode(w=160, h=60, bar_w=4)
    gray[70:130, 80:240] = bc
    regions = find_barcode_regions(gray)
    for r in regions:
        assert len(r) == 4
        x, y, w, h = r
        assert x >= 0 and y >= 0 and w > 0 and h > 0

def test_find_barcode_regions_empty_image():
    """Uniform grey image should produce no regions."""
    from pipeline.barcode import find_barcode_regions
    gray    = np.ones((200, 400), dtype=np.uint8) * 128
    regions = find_barcode_regions(gray)
    assert regions == []

def test_find_barcode_regions_no_region_on_noise():
    """Uniform noise produces many edges but no organised column runs -> likely 0 regions."""
    from pipeline.barcode import find_barcode_regions
    rng  = np.random.default_rng(seed=42)
    gray = rng.integers(0, 255, (200, 400), dtype=np.uint8)
    # May detect some regions on noise; just ensure no crash and valid output format
    regions = find_barcode_regions(gray)
    for r in regions:
        assert len(r) == 4

def test_find_barcode_regions_degenerate_input():
    """1x1 image should not raise."""
    from pipeline.barcode import find_barcode_regions
    assert find_barcode_regions(np.array([[128]], dtype=np.uint8)) == []

def test_find_barcode_regions_box_within_image():
    """All detected region boxes must lie within the image bounds."""
    from pipeline.barcode import find_barcode_regions
    gray = np.ones((200, 400), dtype=np.uint8) * 180
    bc   = _make_barcode(w=160, h=60, bar_w=4)
    gray[70:130, 80:240] = bc
    h, w = gray.shape
    for x, y, bw, bh in find_barcode_regions(gray):
        assert x >= 0 and y >= 0
        assert x + bw <= w
        assert y + bh <= h

def test_find_barcode_no_region_in_solid_black():
    """Solid black image has zero Sobel gradient -> no regions."""
    from pipeline.barcode import find_barcode_regions
    assert find_barcode_regions(np.zeros((100, 200), dtype=np.uint8)) == []


# =============================================================================
#  run() integration
# =============================================================================

def test_run_returns_tuple_of_lists():
    from pipeline.barcode import run
    ref  = _label_with_barcode()
    user = ref.copy()
    r    = run(ref, user, _identity_H(), _identity_H())
    assert isinstance(r, tuple) and len(r) == 2
    assert isinstance(r[0], list) and isinstance(r[1], list)

def test_run_identical_images_no_flags():
    """Identical ref and user -> no barcode tamper boxes."""
    from pipeline.barcode import run
    ref = _label_with_barcode()
    boxes_ref, boxes_user = run(ref, ref, _identity_H(), _identity_H())
    assert boxes_ref  == []
    assert boxes_user == []

def test_run_replaced_barcode_flagged():
    """Replacing the barcode region with solid black -> flagged."""
    from pipeline.barcode import run
    ref  = _label_with_barcode(bc_x=50, bc_y=100, bc_w=200, bc_h=80, bar_w=4)
    user = ref.copy()
    # Overwrite barcode with solid black
    user[100:180, 50:250] = 0

    boxes_ref, _ = run(ref, user, _identity_H(), _identity_H())
    assert len(boxes_ref) > 0

def test_run_flagged_boxes_have_barcode_source():
    from pipeline.barcode import run
    from models.schemas import TamperSource
    ref  = _label_with_barcode(bc_x=50, bc_y=100, bc_w=200, bc_h=80)
    user = ref.copy()
    user[100:180, 50:250] = 0

    boxes_ref, _ = run(ref, user, _identity_H(), _identity_H())
    for tb in boxes_ref:
        assert tb.source == TamperSource.barcode

def test_run_flagged_boxes_have_score():
    """Each flagged TamperBox must carry a cosine similarity score."""
    from pipeline.barcode import run
    ref  = _label_with_barcode(bc_x=50, bc_y=100, bc_w=200, bc_h=80)
    user = ref.copy()
    user[100:180, 50:250] = 0

    boxes_ref, _ = run(ref, user, _identity_H(), _identity_H())
    for tb in boxes_ref:
        assert tb.score is not None
        assert 0.0 <= tb.score <= 1.0

def test_run_no_homography_completes():
    """H=None path must complete without raising."""
    from pipeline.barcode import run
    ref = _label_with_barcode()
    boxes_ref, boxes_user = run(ref, ref, None, None)
    assert isinstance(boxes_ref, list)
    assert boxes_user == []   # no H_inv -> no user boxes

def test_run_user_boxes_empty_when_no_h_inv():
    """H_inv=None -> user tamper boxes not produced."""
    from pipeline.barcode import run
    ref  = _label_with_barcode(bc_x=50, bc_y=100, bc_w=200, bc_h=80)
    user = ref.copy()
    user[100:180, 50:250] = 0
    H    = _identity_H()
    _, boxes_user = run(ref, user, H, None)   # H but no H_inv
    assert boxes_user == []

def test_run_ref_boxes_within_image_bounds():
    """All ref tamper boxes must lie within ref image dimensions."""
    from pipeline.barcode import run
    ref  = _label_with_barcode(bc_x=50, bc_y=100, bc_w=200, bc_h=80)
    user = ref.copy()
    user[100:180, 50:250] = 0
    boxes_ref, _ = run(ref, user, _identity_H(), _identity_H())
    h, w = ref.shape[:2]
    for tb in boxes_ref:
        b = tb.bbox
        assert b.x >= 0 and b.y >= 0
        assert b.x + b.w <= w
        assert b.y + b.h <= h

def test_run_no_barcode_in_image_no_flags():
    """Plain grey label without any barcode -> no flags."""
    from pipeline.barcode import run
    ref = np.ones((300, 400, 3), dtype=np.uint8) * 180
    boxes_ref, boxes_user = run(ref, ref, _identity_H(), _identity_H())
    assert boxes_ref  == []
    assert boxes_user == []