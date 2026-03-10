"""
pipeline/border.py - White border / padding delta detection.

Detects when the user scan has significantly more white padding around the
label content than the reference image — a common sign of image compositing,
re-saving with added white space, or fraudulent substitution of label content.

Checks all four edges independently.  An edge is flagged when ALL of:
  1. The user's normalised border fraction exceeds the ref's by > MIN_BORDER_FRAC.
  2. The absolute border thickness in user pixels exceeds MIN_BORDER_DELTA_PX.
  3. The border strip is at least BORDER_UNIFORMITY_FRAC white pixels.

Why normalise before comparing:
  Ref and user images may be different pixel dimensions (different camera
  distance, different crop).  A ref at 800×1200 and user at 400×600 would
  both have a 20 px left border — identical proportionally, but 20 px raw
  delta if compared directly.  We express every border as a fraction of its
  own image width (or height) so the comparison is resolution-independent.

Thumbnail strategy in content_bounds:
  Finding the content bounding box on a multi-megapixel image is expensive.
  We thumbnail to BOUNDS_THUMB_PX on the longest edge, threshold in greyscale,
  and scale the resulting bbox coordinates back to full resolution.
  Accuracy within ~1/BOUNDS_THUMB_PX of image dimension is sufficient for the
  20 px minimum delta filter.

All functions are synchronous.  No I/O.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import (
    BORDER_UNIFORMITY_FRAC,
    BOUNDS_THUMB_PX,
    MIN_BORDER_DELTA_PX,
    MIN_BORDER_FRAC,
    WHITE_THRESHOLD,
)
from models.schemas import BoundingBox, TamperBox, TamperSource

log = logging.getLogger(__name__)

# Sides enumerated for iteration
_SIDES = ("top", "bottom", "left", "right")


# =============================================================================
#  CONTENT BOUNDING BOX
# =============================================================================

def content_bounds(img_bgr: np.ndarray) -> BoundingBox | None:
    """
    Find the tight bounding box of non-white content in `img_bgr`.

    Operates on a thumbnail (longest edge ≤ BOUNDS_THUMB_PX) for speed,
    then scales coordinates back to full-resolution.

    A pixel is considered white when ALL three BGR channels are ≥ WHITE_THRESHOLD.
    Non-white pixels define the content region.

    Returns:
        BoundingBox in full-resolution pixel coordinates, or None if the
        entire image is white (no content found).
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    # Thumbnail for speed
    scale     = min(1.0, BOUNDS_THUMB_PX / max(h, w))
    thumb_w   = max(1, int(w * scale))
    thumb_h   = max(1, int(h * scale))
    thumb     = cv2.resize(img_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    # White mask: True where ALL channels >= WHITE_THRESHOLD
    white_mask   = np.all(thumb >= WHITE_THRESHOLD, axis=2)   # (thumb_h, thumb_w), bool
    content_mask = ~white_mask                                 # True = non-white content

    rows = np.any(content_mask, axis=1)   # (thumb_h,) — rows that contain content
    cols = np.any(content_mask, axis=0)   # (thumb_w,) — cols that contain content

    if not rows.any():
        return None

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    # Thumbnail bbox
    t_x1 = int(col_indices[0])
    t_y1 = int(row_indices[0])
    t_x2 = int(col_indices[-1]) + 1
    t_y2 = int(row_indices[-1]) + 1

    # Scale back to full resolution — clip to image bounds
    x1 = int(t_x1 / scale)
    y1 = int(t_y1 / scale)
    x2 = min(w, int(t_x2 / scale))
    y2 = min(h, int(t_y2 / scale))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    return BoundingBox(x=x1, y=y1, w=bw, h=bh)


# =============================================================================
#  BORDER UNIFORMITY
# =============================================================================

def measure_border_uniformity(
    img_bgr:   np.ndarray,
    side:      str,
    thickness: int,
) -> float:
    """
    Return the fraction of pixels in a border strip that are white
    (all three BGR channels ≥ WHITE_THRESHOLD).

    Args:
        img_bgr:   Full-resolution image.
        side:      One of "top", "bottom", "left", "right".
        thickness: Strip thickness in pixels.  Must be ≥ 1.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 for degenerate inputs
        (zero-thickness strip, empty image, unknown side).
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0 or thickness < 1:
        return 0.0

    # Clamp thickness to available dimension
    if side == "top":
        strip = img_bgr[:min(thickness, h), :, :]
    elif side == "bottom":
        strip = img_bgr[max(0, h - thickness):, :, :]
    elif side == "left":
        strip = img_bgr[:, :min(thickness, w), :]
    elif side == "right":
        strip = img_bgr[:, max(0, w - thickness):, :]
    else:
        log.warning("measure_border_uniformity: unknown side '%s'", side)
        return 0.0

    if strip.size == 0:
        return 0.0

    white_pixels = np.all(strip >= WHITE_THRESHOLD, axis=2).sum()
    total_pixels = strip.shape[0] * strip.shape[1]

    if total_pixels == 0:
        return 0.0

    return float(white_pixels / total_pixels)


# =============================================================================
#  BORDER WIDTH HELPERS
# =============================================================================

def _border_widths(bounds: BoundingBox, img_h: int, img_w: int) -> dict[str, int]:
    """
    Extract border pixel widths on all four sides from a content bounding box.

    Returns dict with keys "top", "bottom", "left", "right" — all non-negative.
    """
    left   = max(0, bounds.x)
    top    = max(0, bounds.y)
    right  = max(0, img_w - (bounds.x + bounds.w))
    bottom = max(0, img_h - (bounds.y + bounds.h))
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _border_fracs(widths: dict[str, int], img_h: int, img_w: int) -> dict[str, float]:
    """
    Normalise border pixel widths by image dimension.
    Horizontal borders (left/right) normalised by width; vertical by height.
    """
    return {
        "top":    widths["top"]    / max(1, img_h),
        "bottom": widths["bottom"] / max(1, img_h),
        "left":   widths["left"]   / max(1, img_w),
        "right":  widths["right"]  / max(1, img_w),
    }


def _tamper_box_for_side(side: str, thickness: int, img_h: int, img_w: int) -> BoundingBox:
    """
    Build the BoundingBox covering the border strip on `side`.
    In user image coordinates.
    """
    if side == "top":
        return BoundingBox(x=0, y=0, w=img_w, h=thickness)
    elif side == "bottom":
        return BoundingBox(x=0, y=max(0, img_h - thickness), w=img_w, h=thickness)
    elif side == "left":
        return BoundingBox(x=0, y=0, w=thickness, h=img_h)
    else:  # right
        return BoundingBox(x=max(0, img_w - thickness), y=0, w=thickness, h=img_h)


# =============================================================================
#  ORCHESTRATOR
# =============================================================================

def run(
    ref_img:  np.ndarray,
    user_img: np.ndarray,
) -> list[TamperBox]:
    """
    Compare white border widths between ref and user on all four edges.

    For each edge, flag if the user image has significantly more white padding
    than the reference.  All three conditions must hold:
      1. Fractional delta (user_frac - ref_frac) > MIN_BORDER_FRAC.
      2. Absolute user border thickness > MIN_BORDER_DELTA_PX.
      3. Strip uniformity ≥ BORDER_UNIFORMITY_FRAC.

    Returns:
        List of TamperBox in user image coordinates.
        source = TamperSource.white_border.
        score  = normalised fractional delta (user_frac - ref_frac).
    """
    ref_h, ref_w   = ref_img.shape[:2]
    user_h, user_w = user_img.shape[:2]

    # Step 1: content bounding boxes
    ref_bounds  = content_bounds(ref_img)
    user_bounds = content_bounds(user_img)

    if ref_bounds is None:
        log.warning("Border: ref image is entirely white — skipping.")
        return []

    if user_bounds is None:
        log.warning("Border: user image is entirely white — skipping.")
        return []

    # Step 2: border widths and fractions
    ref_widths  = _border_widths(ref_bounds,  ref_h,  ref_w)
    user_widths = _border_widths(user_bounds, user_h, user_w)

    ref_fracs  = _border_fracs(ref_widths,  ref_h,  ref_w)
    user_fracs = _border_fracs(user_widths, user_h, user_w)

    log.debug(
        "Border widths — ref: %s  user: %s",
        {k: f"{v}px/{ref_fracs[k]:.3f}" for k, v in ref_widths.items()},
        {k: f"{v}px/{user_fracs[k]:.3f}" for k, v in user_widths.items()},
    )

    # Step 3: check each side
    flagged: list[TamperBox] = []

    for side in _SIDES:
        delta_frac  = user_fracs[side] - ref_fracs[side]
        user_px     = user_widths[side]

        if delta_frac <= MIN_BORDER_FRAC:
            continue
        if user_px < MIN_BORDER_DELTA_PX:
            continue

        uniformity = measure_border_uniformity(user_img, side, user_px)
        if uniformity < BORDER_UNIFORMITY_FRAC:
            log.debug(
                "Border: %s side delta=%.3f px=%d but uniformity=%.3f < %.3f — skipped.",
                side, delta_frac, user_px, uniformity, BORDER_UNIFORMITY_FRAC,
            )
            continue

        log.info(
            "Border flag: %s  delta_frac=%.3f  user_px=%d  uniformity=%.3f",
            side, delta_frac, user_px, uniformity,
        )

        bbox = _tamper_box_for_side(side, user_px, user_h, user_w)
        flagged.append(TamperBox(
            bbox=bbox,
            source=TamperSource.white_border,
            score=round(delta_frac, 4),
        ))

    log.info("Border complete: %d edge(s) flagged.", len(flagged))
    return flagged