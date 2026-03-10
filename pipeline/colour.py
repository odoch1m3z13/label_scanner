"""
pipeline/colour.py - Text ink colour change detection in LAB colour space.

Operates on DiffEntry pairs (type='match' or 'modified') from diff.py.
Samples ink pixels within each word polygon, computes LAB chroma, and
flags pairs whose calibrated chroma delta exceeds the threshold.

Sampling strategy:
  1. Crop the bounding rect of the word polygon from the image.
  2. Build a tight polygon mask restricting sampling to the word shape.
  3. Run Otsu threshold on the 2D greyscale crop (NOT on a 1D pixel array —
     the 1D form can return thresh=0 on bimodal {0,255} distributions, making
     the < comparison useless). Use cv2.threshold binary output directly.
  4. Ink class = MINORITY pixel class (fewer pixels than background). This
     handles both dark-on-light and light-on-dark labels correctly.
  5. Reject if background is chromatic (LAB chroma > _MAX_BG_CHROMA).
  6. Return mean LAB a*, b* of ink pixels.

Calibration:
  Camera white-balance and scanner colour temperature differ between the
  reference and user images. Without calibration, a systematic +5 unit a*
  shift across all words would flag every word as colour-changed even though
  nothing on the label changed. We compute the median a*/b* offset across
  all matched pairs (median not mean — outlier words can't bias it) and
  subtract it before threshold comparison.

Dynamic threshold (FIX-D from v4.2):
  When homography inlier_ratio > HOMOGRAPHY_INLIER_RATIO_HIGH the images are
  well-aligned. JPEG re-encoding between reference master and re-scanned copy
  introduces consistent ~3-8 LAB unit drift. The threshold is multiplied by
  COLOR_DELTA_HIGH_INLIER_MULTIPLIER to absorb this.

All functions are synchronous, pure (no I/O, no side effects).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import (
    COLOR_DELTA_HIGH_INLIER_MULTIPLIER,
    COLOR_DELTA_THRESHOLD,
    HOMOGRAPHY_INLIER_RATIO_HIGH,
    MIN_BBOX_AREA_FOR_COLOR,
    MIN_BBOX_H_FOR_COLOR,
    MIN_INK_PIXELS,
)
from models.schemas import DiffEntry, DiffType

log = logging.getLogger(__name__)

# Maximum mean LAB chroma sqrt(a*^2 + b*^2) of the BACKGROUND class
# (centred at 0) before a crop is considered to have a coloured background.
# Paper-white and card backgrounds score < 10; vivid packaging scores > 25.
_MAX_BG_CHROMA = 20.0

# Minimum fraction of polygon interior that must be classified as ink.
_MIN_INK_FRAC = 0.05

# Minimum fraction of polygon interior that must be classified as background.
_MIN_BG_FRAC = 0.15


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _crop_word(
    img_bgr: np.ndarray,
    polygon: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Crop the axis-aligned bounding rect of `polygon` from `img_bgr` and
    return (crop_bgr, poly_mask).

    poly_mask is uint8 (255 inside polygon, 0 outside) at crop dimensions.
    Restricts sampling to the actual word shape, important for rotated boxes.

    Returns None when:
      - polygon is empty
      - bounding rect is degenerate (zero area)
      - bounding rect is below MIN_BBOX_* thresholds
      - bounding rect is entirely outside image bounds
    """
    if not polygon:
        return None

    pts = np.array(polygon, dtype=np.int32)
    x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())

    ih, iw = img_bgr.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(iw, x1); y1 = min(ih, y1)

    bw, bh = x1 - x0, y1 - y0
    if bw < 1 or bh < 1:
        return None
    if bw * bh < MIN_BBOX_AREA_FOR_COLOR:
        return None
    if bh < MIN_BBOX_H_FOR_COLOR:
        return None

    crop = img_bgr[y0:y1, x0:x1].copy()

    # Build polygon mask in crop coordinate space
    shifted   = (pts - np.array([x0, y0], dtype=np.int32)).reshape(1, -1, 2)
    poly_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(poly_mask, shifted, 255)

    return crop, poly_mask


# =============================================================================
#  INK SAMPLING
# =============================================================================

def sample_ink_lab(
    img_bgr: np.ndarray,
    polygon: list[tuple[int, int]],
) -> tuple[float, float] | None:
    """
    Sample the dominant ink colour inside a word polygon in LAB (a*, b*) space.

    Returns (mean_a, mean_b) of ink pixels in OpenCV uint8 LAB encoding
    (a* and b* are offset by 128, so neutral grey = 128, not 0).

    Returns None when the region is too small, has insufficient ink pixels,
    has a non-achromatic background, or Otsu produces a degenerate result.

    Critical implementation note:
      We use the BINARY OUTPUT of cv2.threshold (not the threshold value) to
      classify pixels. When the image has a pure bimodal {0, 255} distribution
      OpenCV's Otsu returns thresh=0.0, so the comparison `gray < thresh`
      classifies nothing as dark. Using the binary output mask directly
      (binary==0 for dark, binary==255 for light) works correctly in all cases.
    """
    result = _crop_word(img_bgr, polygon)
    if result is None:
        return None
    crop, poly_mask = result

    # Greyscale for Otsu segmentation — run on full 2D crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Check we have enough interior pixels
    interior_count = int((poly_mask > 0).sum())
    if interior_count < MIN_INK_PIXELS:
        return None

    # Otsu threshold on the 2D grayscale crop.
    # binary: 255 where pixel > thresh (light), 0 where pixel <= thresh (dark).
    # We use the binary output directly — NOT `gray < thresh` — because
    # thresh can be 0.0 on bimodal {0,255} distributions, making < useless.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_mask  = (binary == 0)   & (poly_mask > 0)
    light_mask = (binary == 255) & (poly_mask > 0)

    n_dark  = int(dark_mask.sum())
    n_light = int(light_mask.sum())
    n_total = n_dark + n_light

    if n_total < MIN_INK_PIXELS:
        return None

    # Ink = minority class (fewer pixels than background in any word crop)
    if n_dark <= n_light:
        ink_mask = dark_mask
        bg_mask  = light_mask
    else:
        ink_mask = light_mask
        bg_mask  = dark_mask

    n_ink = int(ink_mask.sum())
    n_bg  = int(bg_mask.sum())

    if n_ink < MIN_INK_PIXELS:
        return None
    if n_ink / n_total < _MIN_INK_FRAC:
        return None
    if n_bg / n_total < _MIN_BG_FRAC:
        return None

    # Reject chromatic backgrounds — convert to LAB and check background chroma
    if n_bg > 0:
        crop_lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
        bg_a      = crop_lab[:, :, 1][bg_mask]   # uint8 LAB: centred at 128
        bg_b      = crop_lab[:, :, 2][bg_mask]
        bg_chroma = float(np.sqrt(((bg_a - 128.0)**2 + (bg_b - 128.0)**2).mean()))
        if bg_chroma > _MAX_BG_CHROMA:
            log.debug(
                "sample_ink_lab: coloured bg (chroma=%.1f > %.1f) — skipped.",
                bg_chroma, _MAX_BG_CHROMA,
            )
            return None

    # Sample ink LAB — a* and b* in uint8 LAB space (neutral = 128)
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
    ink_a    = float(crop_lab[:, :, 1][ink_mask].mean())
    ink_b    = float(crop_lab[:, :, 2][ink_mask].mean())

    return ink_a, ink_b


# =============================================================================
#  CALIBRATION
# =============================================================================

def compute_chroma_calibration(
    diff:     list[DiffEntry],
    ref_img:  np.ndarray,
    user_img: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the median LAB (a*, b*) offset between matched word pairs to
    absorb camera/scanner white-balance drift.

    Only DiffType.match entries are used — modified/removed/added words may
    have genuinely different colours and would bias the offset.

    Returns (a_offset, b_offset).
    Returns (0.0, 0.0) if fewer than 3 matched pairs can be sampled.
    """
    a_deltas: list[float] = []
    b_deltas: list[float] = []

    for entry in diff:
        if entry.type != DiffType.match:
            continue
        if entry.ref is None or entry.user is None:
            continue

        ref_sample  = sample_ink_lab(ref_img,  list(entry.ref.polygon.points))
        user_sample = sample_ink_lab(user_img, list(entry.user.polygon.points))

        if ref_sample is None or user_sample is None:
            continue

        a_deltas.append(user_sample[0] - ref_sample[0])
        b_deltas.append(user_sample[1] - ref_sample[1])

    if len(a_deltas) < 3:
        log.debug(
            "compute_chroma_calibration: %d pairs sampled — zero offset fallback.",
            len(a_deltas),
        )
        return 0.0, 0.0

    a_offset = float(np.median(a_deltas))
    b_offset = float(np.median(b_deltas))

    log.info(
        "Chroma calibration: %d pairs  a_off=%.2f  b_off=%.2f",
        len(a_deltas), a_offset, b_offset,
    )
    return a_offset, b_offset


# =============================================================================
#  COLOUR CHANGE DETECTION
# =============================================================================

def _chroma_delta(
    ref_sample:  tuple[float, float],
    user_sample: tuple[float, float],
    a_offset:    float,
    b_offset:    float,
) -> float:
    """
    Calibrated LAB chroma delta between ref and user ink samples.

    Subtracts the calibration offset from the user sample before computing
    Euclidean distance in (a*, b*) space. Both values are in uint8 LAB
    encoding (neutral = 128), but since we compute differences the 128 offset
    cancels and no re-centring is needed.
    """
    cal_a = user_sample[0] - a_offset
    cal_b = user_sample[1] - b_offset
    da    = ref_sample[0] - cal_a
    db    = ref_sample[1] - cal_b
    return float(np.sqrt(da * da + db * db))


def detect_color_changes(
    diff:         list[DiffEntry],
    ref_img:      np.ndarray,
    user_img:     np.ndarray,
    inlier_ratio: float = 0.0,
) -> tuple[list[DiffEntry], tuple[float, float]]:
    """
    Promote DiffEntry items from 'match' or 'modified' to 'color_changed'
    where the ink colour differs beyond the calibrated threshold.

    Why 'modified' entries are also checked:
      A word can have both a text edit and an ink colour change (e.g. "5g"
      changed to "6g" AND reprinted in red). color_changed supersedes modified
      when colour is the dominant signal; the frontend shows both.

    Dynamic threshold (FIX-D):
      High inlier_ratio means well-aligned images. JPEG re-encoding introduces
      consistent ~3-8 LAB unit drift; the multiplier absorbs this to suppress
      false positives on well-aligned scans.

    Args:
        diff:         Flat list of DiffEntry from diff.run().
        ref_img:      Reference image (BGR, full resolution).
        user_img:     User image (BGR, projected to ref coordinate space).
        inlier_ratio: Homography quality from AlignmentResult.

    Returns:
        (updated_diff, (a_offset, b_offset))
    """
    # Step 1: calibration offset from matched pairs
    a_offset, b_offset = compute_chroma_calibration(diff, ref_img, user_img)

    # Step 2: effective threshold (FIX-D)
    effective_thresh = COLOR_DELTA_THRESHOLD
    if inlier_ratio > HOMOGRAPHY_INLIER_RATIO_HIGH:
        effective_thresh *= COLOR_DELTA_HIGH_INLIER_MULTIPLIER
        log.debug(
            "Colour: high inlier=%.3f — threshold raised to %.1f",
            inlier_ratio, effective_thresh,
        )

    # Step 3–5: per-entry sampling and promotion
    result:        list[DiffEntry] = []
    color_changes: int             = 0

    for entry in diff:
        if entry.type not in (DiffType.match, DiffType.modified):
            result.append(entry)
            continue

        ref_word  = entry.ref
        user_word = entry.user
        if ref_word is None or user_word is None:
            result.append(entry)
            continue

        ref_sample  = sample_ink_lab(ref_img,  list(ref_word.polygon.points))
        user_sample = sample_ink_lab(user_img, list(user_word.polygon.points))

        if ref_sample is None or user_sample is None:
            result.append(entry)
            continue

        delta = _chroma_delta(ref_sample, user_sample, a_offset, b_offset)

        if delta > effective_thresh:
            color_changes += 1
            log.debug(
                "Colour change: '%s' delta=%.1f thresh=%.1f "
                "ref=(%.1f,%.1f) user=(%.1f,%.1f)",
                ref_word.text, delta, effective_thresh,
                ref_sample[0] - 128, ref_sample[1] - 128,
                user_sample[0] - 128, user_sample[1] - 128,
            )
            from models.schemas import DiffEntry as _DE, DiffType as _DT
            result.append(_DE(
                type=_DT.color_changed,
                ref=ref_word,
                user=user_word,
                color_delta=round(delta, 2),
                similarity=entry.similarity,   # preserve text similarity if modified
            ))
        else:
            result.append(entry)

    log.info(
        "detect_color_changes: %d/%d promoted (thresh=%.1f a_off=%.2f b_off=%.2f)",
        color_changes, len(diff), effective_thresh, a_offset, b_offset,
    )
    return result, (a_offset, b_offset)