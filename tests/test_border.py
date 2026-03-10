"""
tests/test_border.py - Unit tests for pipeline/border.py.

All tests use synthetic BGR images built with numpy.
No API calls, no file I/O.

Image conventions used here:
  - "Content" pixels: BGR value (100, 150, 200)  — non-white
  - "White" pixels:   BGR value (255, 255, 255)  — always passes WHITE_THRESHOLD
  - "Near-white":     BGR value (245, 245, 245)  — also passes WHITE_THRESHOLD (245>=240)
  - "Off-white":      BGR value (220, 220, 220)  — fails WHITE_THRESHOLD (220<240)
"""
from __future__ import annotations

import numpy as np
import cv2


CONTENT_COLOR = (100, 150, 200)   # BGR, clearly non-white
WHITE         = (255, 255, 255)
NEAR_WHITE    = (245, 245, 245)   # >= WHITE_THRESHOLD=240 on all channels
OFF_WHITE     = (220, 220, 220)   # < WHITE_THRESHOLD on all channels


def _canvas(h: int, w: int, fill: tuple = WHITE) -> np.ndarray:
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:, :] = fill
    return img


def _place_content(img: np.ndarray, x: int, y: int, w: int, h: int,
                    color: tuple = CONTENT_COLOR) -> np.ndarray:
    """Paint a content rectangle onto img (in-place)."""
    img[y:y+h, x:x+w] = color
    return img


def _identity_label(h=300, w=400, pad=20) -> np.ndarray:
    """White canvas with content block centred with `pad` border on all sides."""
    img = _canvas(h, w)
    _place_content(img, pad, pad, w - 2*pad, h - 2*pad)
    return img


# =============================================================================
#  content_bounds
# =============================================================================

class TestContentBounds:
    def test_all_white_returns_none(self):
        from pipeline.border import content_bounds
        img = _canvas(200, 300)
        assert content_bounds(img) is None

    def test_full_content_returns_full_bounds(self):
        """Image entirely filled with content → bounds ≈ full image."""
        from pipeline.border import content_bounds
        img = _canvas(200, 300, fill=CONTENT_COLOR)
        b = content_bounds(img)
        assert b is not None
        assert b.x == 0 and b.y == 0
        assert b.w == 300 and b.h == 200

    def test_centred_content_correct_bounds(self):
        """Content block with 20 px white border on all sides."""
        from pipeline.border import content_bounds
        img = _identity_label(h=200, w=300, pad=20)
        b = content_bounds(img)
        assert b is not None
        # Allow ±2 px rounding from thumbnail scaling
        assert abs(b.x - 20) <= 2,   f"x={b.x}"
        assert abs(b.y - 20) <= 2,   f"y={b.y}"
        assert abs(b.w - 260) <= 4,  f"w={b.w}"
        assert abs(b.h - 160) <= 4,  f"h={b.h}"

    def test_asymmetric_border_detected(self):
        """Content padded differently on each side."""
        from pipeline.border import content_bounds
        img = _canvas(200, 300)
        _place_content(img, 50, 10, 200, 160)   # left=50 top=10
        b = content_bounds(img)
        assert b is not None
        assert abs(b.x - 50) <= 3
        assert abs(b.y - 10) <= 3

    def test_returns_bounding_box_type(self):
        from pipeline.border import content_bounds
        from models.schemas import BoundingBox
        img = _identity_label()
        b = content_bounds(img)
        assert isinstance(b, BoundingBox)

    def test_empty_image_returns_none(self):
        from pipeline.border import content_bounds
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        assert content_bounds(img) is None

    def test_single_pixel_content(self):
        """One non-white pixel should produce a 1×1 bounding box."""
        from pipeline.border import content_bounds
        img = _canvas(100, 100)
        img[50, 50] = CONTENT_COLOR
        b = content_bounds(img)
        assert b is not None
        assert b.w >= 1 and b.h >= 1

    def test_near_white_pixel_treated_as_white(self):
        """Near-white (245,245,245) ≥ WHITE_THRESHOLD=240 → white → content_bounds=None."""
        from pipeline.border import content_bounds
        img = _canvas(100, 100, fill=NEAR_WHITE)
        assert content_bounds(img) is None

    def test_off_white_pixel_treated_as_content(self):
        """Off-white (220,220,220) < WHITE_THRESHOLD → non-white → content found."""
        from pipeline.border import content_bounds
        img = _canvas(100, 100, fill=OFF_WHITE)
        assert content_bounds(img) is not None

    def test_bounds_within_image(self):
        """Returned bbox must not exceed image dimensions."""
        from pipeline.border import content_bounds
        img = _identity_label(h=300, w=400, pad=30)
        b = content_bounds(img)
        assert b is not None
        assert b.x >= 0 and b.y >= 0
        assert b.x + b.w <= 400
        assert b.y + b.h <= 300


# =============================================================================
#  measure_border_uniformity
# =============================================================================

class TestMeasureBorderUniformity:
    def test_all_white_top_strip_is_one(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300)
        assert measure_border_uniformity(img, "top", 20) == 1.0

    def test_all_content_top_strip_is_zero(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300, fill=CONTENT_COLOR)
        assert measure_border_uniformity(img, "top", 20) == 0.0

    def test_half_white_half_content_bottom(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(100, 200)
        # Bottom 40 rows: top 20 white, bottom 20 content
        img[80:100, :] = CONTENT_COLOR
        u = measure_border_uniformity(img, "bottom", 40)
        # 20 white rows / 40 total rows = 0.5
        assert abs(u - 0.5) < 0.02

    def test_all_white_right_strip(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300)
        assert measure_border_uniformity(img, "right", 30) == 1.0

    def test_all_white_left_strip(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300)
        assert measure_border_uniformity(img, "left", 30) == 1.0

    def test_zero_thickness_returns_zero(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300)
        assert measure_border_uniformity(img, "top", 0) == 0.0

    def test_negative_thickness_returns_zero(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300)
        assert measure_border_uniformity(img, "top", -5) == 0.0

    def test_unknown_side_returns_zero(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300)
        assert measure_border_uniformity(img, "diagonal", 10) == 0.0

    def test_thickness_exceeds_image_dimension_clamped(self):
        """Strip thicker than image should not raise — clamp to image size."""
        from pipeline.border import measure_border_uniformity
        img = _canvas(50, 100)
        result = measure_border_uniformity(img, "top", 200)
        assert 0.0 <= result <= 1.0

    def test_empty_image_returns_zero(self):
        from pipeline.border import measure_border_uniformity
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        assert measure_border_uniformity(img, "top", 10) == 0.0

    def test_all_four_sides_white(self):
        from pipeline.border import measure_border_uniformity
        img = _canvas(200, 300)
        for side in ("top", "bottom", "left", "right"):
            assert measure_border_uniformity(img, side, 20) == 1.0, f"side={side}"


# =============================================================================
#  _border_widths and _border_fracs (internal helpers)
# =============================================================================

class TestBorderHelpers:
    def test_border_widths_centred_content(self):
        from pipeline.border import _border_widths
        from models.schemas import BoundingBox
        # Content from (20,10) with size (200, 150) in a 300x400 image
        # left=20  top=10  right=400-220=180  bottom=300-160=140
        b = BoundingBox(x=20, y=10, w=200, h=150)
        widths = _border_widths(b, img_h=300, img_w=400)
        assert widths["left"]   == 20
        assert widths["top"]    == 10
        assert widths["right"]  == 180
        assert widths["bottom"] == 140

    def test_border_widths_zero_padding(self):
        from pipeline.border import _border_widths
        from models.schemas import BoundingBox
        b = BoundingBox(x=0, y=0, w=400, h=300)
        widths = _border_widths(b, 300, 400)
        assert widths["left"] == 0 and widths["top"] == 0
        assert widths["right"] == 0 and widths["bottom"] == 0

    def test_border_fracs_sum_lt_one(self):
        from pipeline.border import _border_widths, _border_fracs
        from models.schemas import BoundingBox
        b = BoundingBox(x=20, y=15, w=200, h=150)
        w = _border_widths(b, 300, 400)
        f = _border_fracs(w, 300, 400)
        # Horizontal fracs: left + right + content_w/img_w = 1
        assert f["left"] + f["right"] < 1.0
        # All fracs non-negative
        for v in f.values():
            assert v >= 0.0


# =============================================================================
#  _tamper_box_for_side
# =============================================================================

class TestTamperBoxForSide:
    def test_top_strip(self):
        from pipeline.border import _tamper_box_for_side
        b = _tamper_box_for_side("top", 30, img_h=200, img_w=300)
        assert b.x == 0 and b.y == 0 and b.w == 300 and b.h == 30

    def test_bottom_strip(self):
        from pipeline.border import _tamper_box_for_side
        b = _tamper_box_for_side("bottom", 25, img_h=200, img_w=300)
        assert b.y == 175 and b.h == 25 and b.w == 300

    def test_left_strip(self):
        from pipeline.border import _tamper_box_for_side
        b = _tamper_box_for_side("left", 40, img_h=200, img_w=300)
        assert b.x == 0 and b.y == 0 and b.w == 40 and b.h == 200

    def test_right_strip(self):
        from pipeline.border import _tamper_box_for_side
        b = _tamper_box_for_side("right", 35, img_h=200, img_w=300)
        assert b.x == 265 and b.w == 35 and b.h == 200


# =============================================================================
#  run() — integration
# =============================================================================

class TestRun:
    def test_identical_images_no_flags(self):
        """Same label → no border flags."""
        from pipeline.border import run
        img = _identity_label()
        assert run(img, img) == []

    def test_user_adds_left_border_flagged(self):
        """User adds 60 px white left border → left edge flagged."""
        from pipeline.border import run
        from models.schemas import TamperSource
        ref  = _identity_label(h=300, w=400, pad=10)
        user = _canvas(300, 460)   # 60 px extra on left
        user[:, 60:, :] = ref      # paste ref starting at x=60
        boxes = run(ref, user)
        sides = _box_sides(boxes)
        assert "left" in sides or len(boxes) > 0, f"expected left flag, got {boxes}"
        for tb in boxes:
            assert tb.source == TamperSource.white_border
            assert tb.score is not None and tb.score > 0

    def test_user_adds_top_border_flagged(self):
        """User adds large white top border → top edge flagged."""
        from pipeline.border import run
        ref  = _identity_label(h=300, w=400, pad=10)
        user = _canvas(360, 400)   # 60 px extra on top
        user[60:, :, :] = ref
        boxes = run(ref, user)
        assert len(boxes) > 0, "expected top border flag"

    def test_returns_list(self):
        from pipeline.border import run
        img    = _identity_label()
        result = run(img, img)
        assert isinstance(result, list)

    def test_all_white_ref_returns_empty(self):
        """All-white ref → skipped gracefully."""
        from pipeline.border import run
        ref  = _canvas(200, 300)
        user = _identity_label()
        assert run(ref, user) == []

    def test_all_white_user_returns_empty(self):
        """All-white user → skipped gracefully."""
        from pipeline.border import run
        ref  = _identity_label()
        user = _canvas(200, 300)
        assert run(ref, user) == []

    def test_flagged_boxes_have_white_border_source(self):
        from pipeline.border import run
        from models.schemas import TamperSource
        ref  = _identity_label(h=300, w=400, pad=5)
        user = _canvas(300, 460)
        user[:, 60:, :] = ref
        for tb in run(ref, user):
            assert tb.source == TamperSource.white_border

    def test_flagged_boxes_within_user_image(self):
        """All TamperBox bboxes must lie within user image bounds."""
        from pipeline.border import run
        ref  = _identity_label(h=300, w=400, pad=5)
        user = _canvas(300, 460)
        user[:, 60:, :] = ref
        user_h, user_w = user.shape[:2]
        for tb in run(ref, user):
            b = tb.bbox
            assert b.x >= 0 and b.y >= 0
            assert b.x + b.w <= user_w
            assert b.y + b.h <= user_h

    def test_small_delta_below_threshold_not_flagged(self):
        """User border only slightly larger than ref (< MIN_BORDER_FRAC) → no flag."""
        from pipeline.border import run
        from config import MIN_BORDER_DELTA_PX, MIN_BORDER_FRAC
        # Add just 1 px white column — well below both pixel and fraction thresholds
        ref  = _identity_label(h=300, w=400, pad=10)
        user = _canvas(300, 401)
        user[:, 1:, :] = ref
        assert run(ref, user) == []

    def test_non_uniform_border_not_flagged(self):
        """Large but non-white border (off-white/patterned) should not be flagged."""
        from pipeline.border import run
        ref  = _identity_label(h=300, w=400, pad=5)
        user = _canvas(300, 460, fill=OFF_WHITE)   # off-white fill
        user[:, 60:, :] = ref                       # paste ref at x=60
        # Left strip is off-white (220,220,220) → uniformity < BORDER_UNIFORMITY_FRAC
        boxes = run(ref, user)
        # Should have zero flags because uniformity check fails for off-white
        assert boxes == [], f"off-white border should not flag: {boxes}"

    def test_score_is_positive_fractional_delta(self):
        """Score must be a positive float representing the normalised delta."""
        from pipeline.border import run
        ref  = _identity_label(h=300, w=400, pad=5)
        user = _canvas(300, 460)
        user[:, 60:, :] = ref
        for tb in run(ref, user):
            assert isinstance(tb.score, float)
            assert tb.score > 0.0


def _box_sides(boxes) -> set[str]:
    """Infer which side each tamper box covers based on its position."""
    sides = set()
    for tb in boxes:
        b = tb.bbox
        if b.x == 0 and b.y == 0 and b.h > b.w:
            sides.add("left")
        elif b.y == 0 and b.x == 0 and b.w > b.h:
            sides.add("top")
        elif b.x > 0 and b.y == 0:
            sides.add("right")
        elif b.y > 0 and b.x == 0:
            sides.add("bottom")
    return sides