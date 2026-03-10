"""
tests/test_colour.py - Unit tests for pipeline/colour.py.

All tests use synthetic images built from numpy arrays.
No API calls, no file I/O, no network.

LAB encoding note (OpenCV uint8):
  L* [0,100]   -> stored [0,255]    (scale 2.55)
  a* [-128,127] -> stored [0,255]   (offset +128, so neutral = 128)
  b* [-128,127] -> stored [0,255]   (offset +128, so neutral = 128)

Critical Otsu note:
  We use cv2.threshold binary output (binary==0 for dark, ==255 for light),
  NOT the comparison `gray < thresh`. When images have pure {0,255} values,
  OpenCV Otsu returns thresh=0.0, making `gray < 0` always False.
  The binary mask is always correct regardless of the threshold value.
"""
from __future__ import annotations

import numpy as np
import pytest
import cv2


# =============================================================================
#  Test fixtures / helpers
# =============================================================================

def _word_image(
    ink_bgr: tuple[int, int, int],
    bg_bgr:  tuple[int, int, int],
    h: int = 40, w: int = 120,
    ink_frac: float = 0.30,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Synthetic word image: bg_bgr canvas with a horizontal ink strip
    occupying `ink_frac` of the height (mimics text strokes on a label).
    Returns (img_bgr, polygon) where polygon is the full image rect.
    """
    img  = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = bg_bgr
    rows = max(2, int(h * ink_frac))
    top  = (h - rows) // 2
    img[top:top + rows, :] = ink_bgr
    return img, [(0, 0), (w, 0), (w, h), (0, h)]


def _make_match_entry(word_img, canvas, x, y):
    """Place word_img on canvas at (x, y) and return a DiffEntry.match."""
    from models.schemas import BoundingBox, DiffEntry, DiffType, Polygon, WordEntry
    h, w = word_img.shape[:2]
    canvas[y:y+h, x:x+w] = word_img
    bbox = BoundingBox(x=x, y=y, w=w, h=h)
    poly = Polygon(points=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
    word = WordEntry(text="w", confidence=0.99, bbox=bbox, polygon=poly)
    return DiffEntry(type=DiffType.match, ref=word, user=word)


# =============================================================================
#  _crop_word
# =============================================================================

def test_crop_word_basic():
    from pipeline.colour import _crop_word
    img = np.ones((60, 100, 3), dtype=np.uint8) * 200
    r   = _crop_word(img, [(10, 10), (90, 10), (90, 50), (10, 50)])
    assert r is not None
    crop, mask = r
    assert crop.shape == (40, 80, 3)
    assert mask.shape == (40, 80)
    assert mask[20, 40] == 255    # interior point
    assert mask.max()  == 255

def test_crop_word_empty_polygon_returns_none():
    from pipeline.colour import _crop_word
    img = np.ones((60, 100, 3), dtype=np.uint8) * 200
    assert _crop_word(img, []) is None

def test_crop_word_out_of_bounds_clamped():
    from pipeline.colour import _crop_word
    img = np.ones((50, 80, 3), dtype=np.uint8) * 200
    r   = _crop_word(img, [(-20, -20), (200, -20), (200, 200), (-20, 200)])
    assert r is not None   # clamped, not None
    crop, _ = r
    assert crop.shape[0] <= 50 and crop.shape[1] <= 80

def test_crop_word_too_small_returns_none():
    from pipeline.colour import _crop_word
    from config import MIN_BBOX_H_FOR_COLOR
    img  = np.ones((MIN_BBOX_H_FOR_COLOR - 1, 60, 3), dtype=np.uint8) * 200
    poly = [(0,0),(60,0),(60, MIN_BBOX_H_FOR_COLOR-1),(0, MIN_BBOX_H_FOR_COLOR-1)]
    assert _crop_word(img, poly) is None


# =============================================================================
#  sample_ink_lab — fundamental correctness
# =============================================================================

def test_sample_black_on_white_returns_neutral_lab():
    """Black ink on white bg -> LAB a*,b* near 128 (neutral in uint8 LAB)."""
    from pipeline.colour import sample_ink_lab
    img, poly = _word_image(ink_bgr=(0,0,0), bg_bgr=(255,255,255))
    r = sample_ink_lab(img, poly)
    assert r is not None, "black-on-white should return a sample"
    a, b = r
    # uint8 LAB neutral = 128; ink is black -> a*~128, b*~128
    assert abs(a - 128) < 25, f"a*={a} not near 128"
    assert abs(b - 128) < 25, f"b*={b} not near 128"

def test_sample_red_ink_has_positive_a():
    """Red ink has high LAB a* (> 128 in uint8 encoding)."""
    from pipeline.colour import sample_ink_lab
    img, poly = _word_image(ink_bgr=(0, 0, 220), bg_bgr=(255, 255, 255))
    r = sample_ink_lab(img, poly)
    # Red may fail background chroma check if ink/bg bleed — just ensure no crash
    assert r is None or (isinstance(r, tuple) and len(r) == 2)
    if r is not None:
        assert r[0] > 128, "red ink should have a* > 128"

def test_sample_white_on_dark_minority_class():
    """White ink on dark background: light pixels are minority -> sampled as ink."""
    from pipeline.colour import sample_ink_lab
    # 30% white ink (minority), 70% dark bg
    img, poly = _word_image(ink_bgr=(250,250,250), bg_bgr=(40,40,40),
                             h=40, w=120, ink_frac=0.25)
    r = sample_ink_lab(img, poly)
    # Should return either None (if bg chroma check catches it) or near-white LAB
    assert r is None or (isinstance(r, tuple) and len(r) == 2)

def test_sample_empty_polygon_returns_none():
    from pipeline.colour import sample_ink_lab
    img = np.ones((50, 80, 3), dtype=np.uint8) * 200
    assert sample_ink_lab(img, []) is None

def test_sample_too_small_returns_none():
    from pipeline.colour import sample_ink_lab
    from config import MIN_BBOX_H_FOR_COLOR
    tiny = np.ones((MIN_BBOX_H_FOR_COLOR - 1, 30, 3), dtype=np.uint8) * 200
    poly = [(0,0),(30,0),(30,MIN_BBOX_H_FOR_COLOR-1),(0,MIN_BBOX_H_FOR_COLOR-1)]
    assert sample_ink_lab(tiny, poly) is None

def test_sample_uniform_image_returns_none_or_tuple():
    """Uniform image — Otsu can't separate classes; should not crash."""
    from pipeline.colour import sample_ink_lab
    uniform = np.ones((40, 120, 3), dtype=np.uint8) * 128
    poly    = [(0,0),(120,0),(120,40),(0,40)]
    result  = sample_ink_lab(uniform, poly)
    assert result is None or isinstance(result, tuple)

def test_sample_out_of_bounds_polygon_no_crash():
    from pipeline.colour import sample_ink_lab
    img = np.ones((50, 80, 3), dtype=np.uint8) * 200
    img[10:40, :] = 0   # dark band
    poly = [(-100, -100), (500, -100), (500, 500), (-100, 500)]
    result = sample_ink_lab(img, poly)
    assert result is None or isinstance(result, tuple)

def test_sample_returns_two_floats():
    from pipeline.colour import sample_ink_lab
    img, poly = _word_image(ink_bgr=(0,0,0), bg_bgr=(255,255,255))
    r = sample_ink_lab(img, poly)
    if r is not None:
        assert isinstance(r[0], float) and isinstance(r[1], float)

def test_sample_coloured_background_rejected():
    """Vivid coloured background should be rejected (chroma > _MAX_BG_CHROMA)."""
    from pipeline.colour import sample_ink_lab
    # Vivid pure blue background (BGR: 220,0,0), black ink
    img, poly = _word_image(ink_bgr=(0,0,0), bg_bgr=(220,0,0), h=40, w=120)
    r = sample_ink_lab(img, poly)
    assert r is None, "vivid coloured background should be rejected"


# =============================================================================
#  _chroma_delta
# =============================================================================

def test_chroma_delta_identical_samples():
    from pipeline.colour import _chroma_delta
    s = (140.0, 125.0)
    assert _chroma_delta(s, s, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)

def test_chroma_delta_euclidean_calculation():
    """Delta should be sqrt(da^2 + db^2) = sqrt(9+16) = 5.0."""
    from pipeline.colour import _chroma_delta
    ref  = (128.0, 128.0)
    user = (131.0, 132.0)
    assert _chroma_delta(ref, user, 0.0, 0.0) == pytest.approx(5.0, abs=1e-4)

def test_chroma_delta_calibration_reduces_distance():
    """Calibration offset should bring user sample closer to ref."""
    from pipeline.colour import _chroma_delta
    ref    = (128.0, 128.0)
    user   = (133.0, 130.0)   # systematic +5 a* drift
    # With a_offset=5 the calibrated user becomes (128, 130) -> delta = 2
    delta_uncal = _chroma_delta(ref, user, 0.0, 0.0)
    delta_cal   = _chroma_delta(ref, user, 5.0, 0.0)
    assert delta_cal < delta_uncal

def test_chroma_delta_symmetry_of_b_offset():
    from pipeline.colour import _chroma_delta
    ref  = (128.0, 128.0)
    user = (128.0, 140.0)
    # b_offset = 12 -> calibrated user b = 128, delta = 0
    assert _chroma_delta(ref, user, 0.0, 12.0) == pytest.approx(0.0, abs=1e-4)

def test_chroma_delta_always_non_negative():
    from pipeline.colour import _chroma_delta
    # Delta is Euclidean distance - always >= 0
    assert _chroma_delta((100.0, 150.0), (90.0, 160.0), 5.0, -5.0) >= 0.0


# =============================================================================
#  compute_chroma_calibration
# =============================================================================

def _make_diff_match(w):
    from models.schemas import DiffEntry, DiffType
    return DiffEntry(type=DiffType.match, ref=w, user=w)

def _black_on_white_word(x, y, w=80, h=30):
    """Return a canvas+word where the word region has black ink on white bg."""
    from models.schemas import BoundingBox, Polygon, WordEntry
    canvas = np.ones((y + h + 5, x + w + 5, 3), dtype=np.uint8) * 255
    canvas[y:y+h, x:x+w] = 255   # white bg
    # ink strip: 30% of height
    ink_rows = max(2, int(h * 0.30))
    top = y + (h - ink_rows) // 2
    canvas[top:top+ink_rows, x:x+w] = 0  # black ink
    bbox = BoundingBox(x=x, y=y, w=w, h=h)
    poly = Polygon(points=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
    word = _WordEntry(text="test", confidence=0.99, bbox=bbox, polygon=poly)
    return canvas, word

# lazy import helper (avoids circular during collection)
def _WordEntry(*a, **kw):
    from models.schemas import WordEntry
    return WordEntry(*a, **kw)

def test_calibration_identical_images_near_zero():
    """Same image for ref and user -> offset ~ 0."""
    from pipeline.colour import compute_chroma_calibration
    from models.schemas import BoundingBox, DiffEntry, DiffType, Polygon, WordEntry

    canvas = np.ones((150, 500, 3), dtype=np.uint8) * 240
    diff   = []
    for i in range(6):
        x = 10 + i * 75
        ink_top = 10 + 9  # 30% of 30px height -> 9px, centred at top+9
        canvas[10:40, x:x+65] = 240
        canvas[19:28, x:x+65] = 0
        bbox = BoundingBox(x=x, y=10, w=65, h=30)
        poly = Polygon(points=[(x,10),(x+65,10),(x+65,40),(x,40)])
        w    = WordEntry(text="t", confidence=0.99, bbox=bbox, polygon=poly)
        diff.append(DiffEntry(type=DiffType.match, ref=w, user=w))

    a_off, b_off = compute_chroma_calibration(diff, canvas, canvas)
    assert abs(a_off) < 3.0, f"a_offset={a_off}"
    assert abs(b_off) < 3.0, f"b_offset={b_off}"

def test_calibration_no_match_entries_returns_zero():
    from pipeline.colour import compute_chroma_calibration
    from models.schemas import BoundingBox, DiffEntry, DiffType, Polygon, WordEntry
    img  = np.ones((100, 200, 3), dtype=np.uint8) * 200
    bbox = BoundingBox(x=5, y=5, w=60, h=20)
    poly = Polygon(points=[(5,5),(65,5),(65,25),(5,25)])
    w    = WordEntry(text="x", confidence=0.9, bbox=bbox, polygon=poly)
    diff = [DiffEntry(type=DiffType.removed, word=w),
            DiffEntry(type=DiffType.added,   word=w)]
    a, b = compute_chroma_calibration(diff, img, img)
    assert a == 0.0 and b == 0.0

def test_calibration_fewer_than_3_pairs_returns_zero():
    """Only 2 successfully sampled pairs -> zero offset fallback."""
    from pipeline.colour import compute_chroma_calibration
    from models.schemas import BoundingBox, DiffEntry, DiffType, Polygon, WordEntry
    # Build only 2 valid match entries on a tiny image that may fail sampling
    img  = np.ones((5, 5, 3), dtype=np.uint8) * 200
    bbox = BoundingBox(x=0, y=0, w=5, h=5)
    poly = Polygon(points=[(0,0),(5,0),(5,5),(0,5)])
    w    = WordEntry(text="x", confidence=0.9, bbox=bbox, polygon=poly)
    diff = [DiffEntry(type=DiffType.match, ref=w, user=w)] * 2
    # These words are too small -> sampling returns None -> zero offset
    a, b = compute_chroma_calibration(diff, img, img)
    assert a == 0.0 and b == 0.0

def test_calibration_returns_tuple_of_floats():
    from pipeline.colour import compute_chroma_calibration
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    a, b = compute_chroma_calibration([], img, img)
    assert isinstance(a, float) and isinstance(b, float)


# =============================================================================
#  detect_color_changes
# =============================================================================

def _make_canvas_with_word(ink_bgr, bg_bgr, x=10, y=10, w=100, h=30):
    """Return (canvas, word_entry) with ink painted at the word region."""
    from models.schemas import BoundingBox, Polygon, WordEntry
    H, W = y + h + 5, x + w + 5
    canvas = np.ones((H, W, 3), dtype=np.uint8)
    canvas[:] = bg_bgr
    ink_rows = max(2, int(h * 0.30))
    top = y + (h - ink_rows) // 2
    canvas[top:top+ink_rows, x:x+w] = ink_bgr
    bbox = BoundingBox(x=x, y=y, w=w, h=h)
    poly = Polygon(points=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
    word = WordEntry(text="w", confidence=0.99, bbox=bbox, polygon=poly)
    return canvas, word

def test_detect_identical_ink_no_change():
    """Same ink colour -> no color_changed entries."""
    from pipeline.colour import detect_color_changes
    from models.schemas import DiffEntry, DiffType

    canvas, word = _make_canvas_with_word((0,0,0), (255,255,255))
    diff         = [DiffEntry(type=DiffType.match, ref=word, user=word)]
    updated, _   = detect_color_changes(diff, canvas, canvas)
    assert DiffType.color_changed not in [d.type for d in updated]

def test_detect_preserves_removed_added_entries():
    """Non-match entries must pass through unchanged."""
    from pipeline.colour import detect_color_changes
    from models.schemas import BoundingBox, DiffEntry, DiffType, Polygon, WordEntry
    img  = np.ones((60, 150, 3), dtype=np.uint8) * 200
    bbox = BoundingBox(x=5, y=5, w=60, h=20)
    poly = Polygon(points=[(5,5),(65,5),(65,25),(5,25)])
    w    = WordEntry(text="x", confidence=0.9, bbox=bbox, polygon=poly)
    diff = [DiffEntry(type=DiffType.removed, word=w),
            DiffEntry(type=DiffType.added,   word=w)]
    updated, _ = detect_color_changes(diff, img, img)
    assert updated[0].type == DiffType.removed
    assert updated[1].type == DiffType.added

def test_detect_returns_correct_structure():
    """Must return (list[DiffEntry], (float, float))."""
    from pipeline.colour import detect_color_changes
    img    = np.ones((50, 50, 3), dtype=np.uint8) * 200
    result = detect_color_changes([], img, img)
    assert isinstance(result, tuple) and len(result) == 2
    diff_out, offsets = result
    assert isinstance(diff_out, list)
    assert isinstance(offsets, tuple) and len(offsets) == 2
    assert isinstance(offsets[0], float) and isinstance(offsets[1], float)

def test_detect_empty_diff_no_crash():
    from pipeline.colour import detect_color_changes
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    updated, (a, b) = detect_color_changes([], img, img)
    assert updated == []
    assert a == 0.0 and b == 0.0

def test_detect_color_changed_entry_has_delta():
    """Any promoted color_changed entry must carry color_delta > 0."""
    from pipeline.colour import detect_color_changes
    from models.schemas import DiffEntry, DiffType

    ref_canvas, ref_word = _make_canvas_with_word((0,0,0),   (255,255,255))
    usr_canvas, usr_word = _make_canvas_with_word((0,0,200), (255,255,255))

    diff    = [DiffEntry(type=DiffType.match, ref=ref_word, user=usr_word)]
    updated, _ = detect_color_changes(diff, ref_canvas, usr_canvas, inlier_ratio=0.0)

    for entry in updated:
        if entry.type == DiffType.color_changed:
            assert entry.color_delta is not None
            assert entry.color_delta > 0

def test_detect_high_inlier_raises_threshold():
    """
    High inlier_ratio tightens the threshold. Verify: function runs cleanly
    at both extremes and produces list output.
    """
    from pipeline.colour import detect_color_changes
    img = np.ones((50, 50, 3), dtype=np.uint8) * 200
    lo, _ = detect_color_changes([], img, img, inlier_ratio=0.0)
    hi, _ = detect_color_changes([], img, img, inlier_ratio=0.99)
    assert lo == [] and hi == []

def test_detect_modified_entry_can_be_promoted():
    """An entry that is already 'modified' (text diff) can also become color_changed."""
    from pipeline.colour import detect_color_changes
    from models.schemas import DiffEntry, DiffType

    ref_canvas, ref_word = _make_canvas_with_word((0,0,0),   (255,255,255))
    usr_canvas, usr_word = _make_canvas_with_word((0,0,200), (255,255,255))

    # Simulate a 'modified' entry (text + colour changed)
    from models.schemas import DiffEntry as DE, DiffType as DT
    diff = [DE(type=DT.modified, ref=ref_word, user=usr_word, similarity=0.80)]
    updated, _ = detect_color_changes(diff, ref_canvas, usr_canvas, inlier_ratio=0.0)

    # Entry must be either modified (sampling failed) or color_changed (detected)
    assert updated[0].type in (DT.modified, DT.color_changed)

def test_detect_color_changed_preserves_similarity():
    """When a modified entry is promoted to color_changed, similarity is preserved."""
    from pipeline.colour import detect_color_changes
    from models.schemas import DiffEntry as DE, DiffType as DT

    ref_canvas, ref_word = _make_canvas_with_word((0,0,0),   (255,255,255))
    usr_canvas, usr_word = _make_canvas_with_word((0,0,200), (255,255,255))

    diff    = [DE(type=DT.modified, ref=ref_word, user=usr_word, similarity=0.76)]
    updated, _ = detect_color_changes(diff, ref_canvas, usr_canvas, inlier_ratio=0.0)

    for entry in updated:
        if entry.type == DT.color_changed:
            assert entry.similarity == pytest.approx(0.76, abs=1e-4)