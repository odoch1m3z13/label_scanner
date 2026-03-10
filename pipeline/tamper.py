"""
pipeline/tamper.py - Masked visual tamper detection.

Workflow:
  1. build_text_mask()  - draw filled rectangles over all Cloud Vision text
                          boxes, leaving only graphic/background regions exposed.
  2. run_visual_diff()  - two complementary methods on the unmasked canvas:
       Method A: FIX-A aware per-pixel diff in LAB space.
                 Near-black pixels get full luminance weight so solid black
                 overlays are never absorbed into the background.
       Method B: Cell-level SSIM - structural similarity per SSIM_CELL_SIZE
                 block. Catches subtle graphic swaps that pass the pixel diff
                 threshold (logo colour shift, ingredient icon swap).

Why text masking uses Cloud Vision boxes (not match-polygon heuristic):
  The v4.2 FIX-B approach only masked polygons confirmed as 'match' by diff.
  This left removed/added words unmasked - those very words are the highest-
  contrast diff regions and triggered false tamper boxes over text changes.
  Cloud Vision gives ALL text boxes before the diff runs so we mask everything
  that is text regardless of whether it changed.

Valid-pixel mask:
  After warpPerspective, pixels outside the user image boundary are filled
  black (0,0,0). We resolve ambiguity with genuine black label pixels by
  warping an all-white canvas with the same H - pixels that stay 0 are fill.

All functions are synchronous (CPU-bound). No I/O, no network calls.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import (
    DARK_PIXEL_L_THRESH,
    GRAPHIC_CELL_FRAC,
    GRAPHIC_DIFF_THRESH,
    GRAPHIC_MIN_AREA_FRAC,
    MAX_TAMPER_AREA_FRAC,
    MIN_TAMPER_AREA_FRAC,
    SSIM_CELL_SIZE,
    SSIM_TAMPER_THRESHOLD,
    TAMPER_COLOR_THRESH,
    TAMPER_MAX_ASPECT_RATIO,
    TAMPER_WORK_SIZE,
    TAMPER_ZONE_PAD_FRAC,
)
from models.schemas import BoundingBox, TamperBox, TamperSource

log = logging.getLogger(__name__)

# Minimum fraction of a cell that must be unmasked (not text, not fill)
# before SSIM is computed for that cell.
_SSIM_MIN_UNMASKED_FRAC = 0.25

# SSIM numerical stability constants (standard values, L=255).
_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2

# Minimum contour area in pixels before the area-fraction filter is applied.
_MIN_CONTOUR_PX = 4


# =============================================================================
#  TEXT MASK
# =============================================================================

def build_text_mask(
    canvas_h:   int,
    canvas_w:   int,
    text_boxes: list[BoundingBox],
    pad:        int = 6,
) -> np.ndarray:
    """
    Build a binary mask from Cloud Vision word bounding boxes.

    255 = text region  (excluded from visual diff)
    0   = graphic/background (included in visual diff)

    Args:
        canvas_h:   Working image height in pixels.
        canvas_w:   Working image width in pixels.
        text_boxes: Word BoundingBoxes in working-image coordinate space.
        pad:        Extra pixels of padding around each box to absorb the
                    gap between the OCR bounding box edge and ink boundary.

    Returns:
        uint8 numpy array of shape (canvas_h, canvas_w).
    """
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for box in text_boxes:
        x1 = max(0,        box.x - pad)
        y1 = max(0,        box.y - pad)
        x2 = min(canvas_w, box.x + box.w + pad)
        y2 = min(canvas_h, box.y + box.h + pad)
        mask[y1:y2, x1:x2] = 255
    return mask


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _cell_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute SSIM between two single-channel cell crops (float32).
    Returns a value in [-1, 1] where 1.0 = identical.
    Uses the standard SSIM formula with constants C1, C2 for numerical
    stability in near-uniform regions.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mu_a   = float(a.mean())
    mu_b   = float(b.mean())
    sig_a  = float(a.std())
    sig_b  = float(b.std())
    sig_ab = float(np.mean((a - mu_a) * (b - mu_b)))

    num = (2.0 * mu_a * mu_b + _SSIM_C1) * (2.0 * sig_ab + _SSIM_C2)
    den = (mu_a**2 + mu_b**2 + _SSIM_C1) * (sig_a**2 + sig_b**2 + _SSIM_C2)
    return float(num / den) if den > 1e-10 else 1.0


def _build_valid_mask(
    user_work: np.ndarray,
    H_work:    np.ndarray,
    work_h:    int,
    work_w:    int,
) -> np.ndarray:
    """
    Build a binary mask indicating which pixels in the warped user image are
    valid (came from the user image) vs fill (black border from warpPerspective).

    Method: warp an all-white canvas; any pixel that stays 0 is fill.

    Returns:
        uint8 array of shape (work_h, work_w). 255=valid, 0=fill.
    """
    white  = np.ones_like(user_work, dtype=np.uint8) * 255
    warped = cv2.warpPerspective(
        white, H_work, (work_w, work_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return (gray > 0).astype(np.uint8) * 255


def _fix_a_diff_map(
    ref_lab:    np.ndarray,
    user_lab:   np.ndarray,
    text_mask:  np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute a per-pixel difference map in LAB space with FIX-A black
    rectangle boosting.

    Standard LAB diff:
        diff = sqrt(dL^2 + da^2 + db^2)

    FIX-A: For pixels where EITHER image has L* < DARK_PIXEL_L_THRESH,
    replace the diff value with 255 (maximum). This ensures solid black
    overlays that produce low absolute deltas vs dark backgrounds are
    always flagged.

    Text-masked and fill pixels are zeroed so they don't contribute to
    contour finding.

    Returns float32 diff map, shape (h, w), values in [0, 255].
    """
    l_ref  = ref_lab [:, :, 0].astype(np.float32)
    l_user = user_lab[:, :, 0].astype(np.float32)
    a_diff = ref_lab [:, :, 1].astype(np.float32) - user_lab[:, :, 1].astype(np.float32)
    b_diff = ref_lab [:, :, 2].astype(np.float32) - user_lab[:, :, 2].astype(np.float32)
    l_diff = l_ref - l_user

    diff = np.sqrt(l_diff**2 + a_diff**2 + b_diff**2)

    # FIX-A: near-black pixels on either image -> maximum weight
    near_black = (l_ref < DARK_PIXEL_L_THRESH) | (l_user < DARK_PIXEL_L_THRESH)
    diff[near_black] = 255.0

    # Zero excluded regions
    diff[text_mask  > 0] = 0.0
    diff[valid_mask == 0] = 0.0

    return diff


def _ssim_flag_map(
    ref_gray:   np.ndarray,
    user_gray:  np.ndarray,
    text_mask:  np.ndarray,
    valid_mask: np.ndarray,
    cell_size:  int,
) -> np.ndarray:
    """
    Build a binary flag map (uint8) by computing per-cell SSIM.

    A cell is flagged (255) when:
      - SSIM < SSIM_TAMPER_THRESHOLD, AND
      - >= _SSIM_MIN_UNMASKED_FRAC of the cell is unmasked (not text, not fill).

    Returns uint8 array of shape (h, w). 255=cell flagged, 0=cell OK.
    """
    h, w   = ref_gray.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            y2 = min(y + cell_size, h)
            x2 = min(x + cell_size, w)

            cell_mask  = text_mask [y:y2, x:x2]
            cell_valid = valid_mask[y:y2, x:x2]
            excluded   = (cell_mask > 0) | (cell_valid == 0)

            unmasked_frac = 1.0 - float(excluded.mean())
            if unmasked_frac < _SSIM_MIN_UNMASKED_FRAC:
                continue

            cell_ref  = ref_gray [y:y2, x:x2].astype(np.float32).copy()
            cell_user = user_gray[y:y2, x:x2].astype(np.float32).copy()
            cell_ref [excluded] = 0.0
            cell_user[excluded] = 0.0

            ssim = _cell_ssim(cell_ref, cell_user)
            if ssim < SSIM_TAMPER_THRESHOLD:
                result[y:y2, x:x2] = 255

    return result


def _contours_to_boxes(
    binary_map: np.ndarray,
    canvas_h:   int,
    canvas_w:   int,
) -> list[BoundingBox]:
    """
    Find contours in a binary map and return filtered BoundingBoxes.

    Filters (all in working-image coordinate space):
      - Area:   MIN_TAMPER_AREA_FRAC <= area/total <= MAX_TAMPER_AREA_FRAC
      - Aspect: max(w/h, h/w) <= TAMPER_MAX_ASPECT_RATIO
      - Min px: area >= _MIN_CONTOUR_PX

    Morphological close is applied first to merge nearby flagged cells.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = max(1, canvas_h * canvas_w)
    boxes: list[BoundingBox] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_CONTOUR_PX:
            continue

        area_frac = area / total_area
        if area_frac < MIN_TAMPER_AREA_FRAC or area_frac > MAX_TAMPER_AREA_FRAC:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 1 or bh < 1:
            continue

        aspect = max(bw / bh, bh / bw)
        if aspect > TAMPER_MAX_ASPECT_RATIO:
            continue

        boxes.append(BoundingBox(x=x, y=y, w=bw, h=bh))

    return boxes


def _scale_boxes(boxes: list[BoundingBox], scale: float) -> list[BoundingBox]:
    """Scale BoundingBoxes from working-image space to full-resolution space."""
    if scale == 1.0:
        return boxes
    return [
        BoundingBox(
            x=int(b.x / scale), y=int(b.y / scale),
            w=max(1, int(b.w / scale)), h=max(1, int(b.h / scale)),
        )
        for b in boxes
    ]


def _scale_text_boxes_to_work(
    boxes: list[BoundingBox], scale: float, work_h: int, work_w: int,
) -> list[BoundingBox]:
    """Scale full-resolution word boxes down to working-image space."""
    if scale == 1.0:
        return boxes
    return [
        BoundingBox(
            x=max(0, int(b.x * scale)), y=max(0, int(b.y * scale)),
            w=max(1, int(b.w * scale)), h=max(1, int(b.h * scale)),
        )
        for b in boxes
    ]


def _project_boxes_to_user(
    boxes:  list[BoundingBox],
    H_inv:  np.ndarray,
    user_h: int,
    user_w: int,
) -> list[BoundingBox]:
    """
    Project BoundingBoxes from ref coordinate space into user coordinate space.
    H_inv maps ref->user (inverse of the user->ref homography).
    All four corners are projected; result is the axis-aligned bounding rect,
    clipped to user image bounds.
    """
    projected: list[BoundingBox] = []
    for box in boxes:
        corners = np.float32([
            [box.x,         box.y         ],
            [box.x + box.w, box.y         ],
            [box.x + box.w, box.y + box.h ],
            [box.x,         box.y + box.h ],
        ]).reshape(-1, 1, 2)

        warped = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
        xs = np.clip(warped[:, 0], 0, user_w)
        ys = np.clip(warped[:, 1], 0, user_h)

        projected.append(BoundingBox(
            x=int(xs.min()), y=int(ys.min()),
            w=max(1, int(xs.max() - xs.min())),
            h=max(1, int(ys.max() - ys.min())),
        ))
    return projected


# =============================================================================
#  VISUAL DIFF  (public, used by tests directly)
# =============================================================================

def run_visual_diff(
    ref_work:     np.ndarray,
    user_aligned: np.ndarray,
    text_mask:    np.ndarray,
    valid_mask:   np.ndarray | None = None,
) -> list[BoundingBox]:
    """
    Detect visually tampered regions in graphic/background areas.

    Method A - FIX-A aware LAB pixel diff:
        Per-pixel Euclidean colour distance in LAB space. Near-black pixels
        (L* < DARK_PIXEL_L_THRESH) on either image are boosted to 255 so
        solid black overlays always flag regardless of background colour.

    Method B - Cell-level SSIM:
        SSIM_CELL_SIZE x SSIM_CELL_SIZE cells checked for structural similarity.
        Catches subtle graphic swaps that pass the pixel diff threshold.

    Both maps are unioned; contours are filtered by area and aspect ratio.

    Args:
        ref_work:     Reference working image (BGR, uint8).
        user_aligned: User image in ref coordinate space (BGR, uint8, same size).
        text_mask:    255=text region (excluded). Shape (h, w).
        valid_mask:   255=valid warped pixel. None->all pixels valid.

    Returns:
        List of BoundingBox in working-image pixel coordinates.
    """
    h, w = ref_work.shape[:2]

    if valid_mask is None:
        valid_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Method A
    ref_lab  = cv2.cvtColor(ref_work,     cv2.COLOR_BGR2Lab)
    user_lab = cv2.cvtColor(user_aligned, cv2.COLOR_BGR2Lab)
    diff_map = _fix_a_diff_map(ref_lab, user_lab, text_mask, valid_mask)
    _, pixel_flags = cv2.threshold(
        diff_map.astype(np.uint8), int(TAMPER_COLOR_THRESH), 255, cv2.THRESH_BINARY
    )

    # Method B
    ref_gray  = cv2.cvtColor(ref_work,     cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_aligned, cv2.COLOR_BGR2GRAY)
    ssim_flags = _ssim_flag_map(ref_gray, user_gray, text_mask, valid_mask, SSIM_CELL_SIZE)

    combined = cv2.bitwise_or(pixel_flags, ssim_flags)
    boxes    = _contours_to_boxes(combined, h, w)

    log.debug("run_visual_diff: %d tamper boxes.", len(boxes))
    return boxes


# =============================================================================
#  ORCHESTRATOR
# =============================================================================

def run(
    ref_img:      np.ndarray,
    user_img:     np.ndarray,
    H:            list[float] | None,
    H_inv:        list[float] | None,
    ref_semantic: object,
    inlier_ratio: float,
) -> tuple[list[TamperBox], list[TamperBox]]:
    """
    Orchestrate the full visual tamper detection pipeline.

    Steps:
      1. Shrink ref to TAMPER_WORK_SIZE working canvas.
      2a. H available: warp user to ref space; build valid-pixel mask.
      2b. H unavailable: resize user to same working dims; no valid mask needed.
      3. Scale ref word boxes to working-image space.
      4. Build text mask from scaled boxes.
      5. Run Method A + B visual diff.
      6. Scale boxes back to full-resolution ref coordinates.
      7. Project full-res ref boxes into user coordinate space via H_inv.

    Args:
        ref_img:      Full-resolution reference image (BGR).
        user_img:     Full-resolution user scan (BGR).
        H:            Flat 9-element homography (user->ref) or None.
        H_inv:        Flat 9-element inverse homography (ref->user) or None.
        ref_semantic: SemanticMap for the reference image (provides word bboxes).
        inlier_ratio: Alignment quality (reserved for adaptive threshold tuning).

    Returns:
        (tamper_boxes_ref, tamper_boxes_user)
        Each TamperBox has source=TamperSource.graphic.
        tamper_boxes_user is empty when H_inv is None.
    """
    ref_h, ref_w   = ref_img.shape[:2]
    user_h, user_w = user_img.shape[:2]

    # Step 1: shrink ref to working size
    ref_scale = min(1.0, TAMPER_WORK_SIZE / max(ref_h, ref_w))
    work_w    = max(1, int(ref_w * ref_scale))
    work_h    = max(1, int(ref_h * ref_scale))
    ref_work  = cv2.resize(ref_img, (work_w, work_h), interpolation=cv2.INTER_AREA)

    # Step 2: aligned user working image
    if H is not None:
        H_mat      = np.array(H, dtype=np.float64).reshape(3, 3)
        user_scale = min(1.0, TAMPER_WORK_SIZE / max(user_h, user_w))
        user_work_w = max(1, int(user_w * user_scale))
        user_work_h = max(1, int(user_h * user_scale))
        user_work   = cv2.resize(user_img, (user_work_w, user_work_h), interpolation=cv2.INTER_AREA)

        # Scale H into working-image space:
        # H_work = S_ref @ H @ S_user_inv
        S_ref      = np.diag([ref_scale,  ref_scale,  1.0])
        S_user_inv = np.diag([1/user_scale, 1/user_scale, 1.0])
        H_work     = S_ref @ H_mat @ S_user_inv

        user_aligned = cv2.warpPerspective(
            user_work, H_work, (work_w, work_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        valid_mask = _build_valid_mask(user_work, H_work, work_h, work_w)
    else:
        user_aligned = cv2.resize(user_img, (work_w, work_h), interpolation=cv2.INTER_AREA)
        valid_mask   = np.ones((work_h, work_w), dtype=np.uint8) * 255
        log.warning("Tamper: no homography - running unaligned diff (higher FP rate).")

    # Step 3 & 4: text mask
    word_boxes_full = [w.bbox for w in getattr(ref_semantic, "words", [])]
    word_boxes_work = _scale_text_boxes_to_work(word_boxes_full, ref_scale, work_h, work_w)
    text_mask       = build_text_mask(work_h, work_w, word_boxes_work, pad=6)

    log.debug(
        "Tamper: work=%dx%d text_boxes=%d mask_coverage=%.1f%%",
        work_w, work_h, len(word_boxes_work),
        100.0 * float((text_mask > 0).mean()),
    )

    # Step 5: visual diff
    work_boxes = run_visual_diff(ref_work, user_aligned, text_mask, valid_mask)
    log.info("Tamper: %d boxes at work resolution.", len(work_boxes))

    # Step 6: scale to full-res ref space
    ref_boxes_full = _scale_boxes(work_boxes, ref_scale)

    # Step 7: project to user space
    user_boxes_full: list[BoundingBox] = []
    if H_inv is not None and ref_boxes_full:
        H_inv_mat       = np.array(H_inv, dtype=np.float64).reshape(3, 3)
        user_boxes_full = _project_boxes_to_user(ref_boxes_full, H_inv_mat, user_h, user_w)

    tamper_ref  = [TamperBox(bbox=b, source=TamperSource.graphic) for b in ref_boxes_full]
    tamper_user = [TamperBox(bbox=b, source=TamperSource.graphic) for b in user_boxes_full]

    log.info("Tamper complete: %d ref, %d user boxes.", len(tamper_ref), len(tamper_user))
    return tamper_ref, tamper_user