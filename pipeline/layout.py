"""
pipeline/layout.py - Layout figure region comparison via IoU matching.

Compares LayoutRegion lists between the reference SemanticMap and the user
SemanticMap (both produced by workers/cloud_vision.py) to detect structural
changes: figures removed, added, or repositioned on the label.

Scope — figure regions only:
  Cloud Vision classifies regions as TEXT, PICTURE, TABLE, BARCODE, etc.
  Layout.py operates exclusively on RegionType.figure (PICTURE blocks).
  Text changes are handled by pipeline/diff.py.
  Barcode changes are handled by pipeline/barcode.py.
  Pixel-level graphic changes within matched figures are caught by
  pipeline/tamper.py's SSIM pass — layout.py handles only structural mismatches.

IoU normalisation:
  Ref and user images may have different pixel dimensions (different camera
  distances, crop ratios). Raw pixel IoU would compare the wrong coordinate
  spaces. We normalise each box to its own image's [0,1]² space before
  computing intersection — this makes the comparison scale-invariant.

Matching strategy (greedy highest-IoU):
  For small N (typically 2–8 figure regions per label) greedy nearest-IoU
  is equivalent to the optimal Hungarian assignment. For large layouts where
  this matters, tamper.py's SSIM will catch any residual errors.

  1. For each ref figure, find the highest-IoU user figure.
  2. IoU >= LAYOUT_IOU_THRESHOLD → matched (structural position preserved).
  3. IoU <  LAYOUT_IOU_THRESHOLD → ref figure unmatched → flag in ref space.
  4. User figures that were never matched → flag in user space.

Output convention:
  tamper_boxes_ref  — flags drawn on the reference image overlay.
  tamper_boxes_user — flags drawn on the user image overlay.
  Unmatched ref boxes are also projected to user space via H_inv so the
  frontend can highlight "what is missing" on both sides simultaneously.
  Unmatched user boxes are already in user space (no projection needed).

All functions are pure (no I/O, no image data).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import LAYOUT_IOU_THRESHOLD
from models.schemas import BoundingBox, LayoutRegion, RegionType, TamperBox, TamperSource

log = logging.getLogger(__name__)


# =============================================================================
#  IoU UTILITIES
# =============================================================================

def _norm_box(b: BoundingBox, img_h: int, img_w: int) -> tuple[float, float, float, float]:
    """
    Normalise a BoundingBox to [0, 1]² space.
    Returns (x1, y1, x2, y2) where all values are in [0, 1].
    Clamps to image bounds to handle boxes that extend slightly outside.
    """
    x1 = max(0.0, b.x             / img_w)
    y1 = max(0.0, b.y             / img_h)
    x2 = min(1.0, (b.x + b.w)    / img_w)
    y2 = min(1.0, (b.y + b.h)    / img_h)
    return x1, y1, x2, y2


def iou(a: BoundingBox, b: BoundingBox, img_h: int, img_w: int) -> float:
    """
    Compute IoU between two BoundingBoxes normalised to image dimensions.

    Both boxes are assumed to be in the same pixel coordinate space
    (img_h × img_w). Normalisation makes the comparison scale-invariant
    so that a logo at (100, 100, 200, 150) in a 1000×800 image is treated
    as equivalent to the same logo at (50, 50, 100, 75) in a 500×400 image.

    Returns a float in [0.0, 1.0]. Returns 0.0 for degenerate boxes (zero area).

    Args:
        a, b:         BoundingBoxes in the same pixel coordinate space.
        img_h, img_w: Image dimensions used for normalisation.
    """
    ax1, ay1, ax2, ay2 = _norm_box(a, img_h, img_w)
    bx1, by1, bx2, by2 = _norm_box(b, img_h, img_w)

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter

    if union < 1e-9:
        return 0.0

    return float(inter / union)


def _cross_iou(
    a: BoundingBox, img_h_a: int, img_w_a: int,
    b: BoundingBox, img_h_b: int, img_w_b: int,
) -> float:
    """
    IoU between two boxes in DIFFERENT coordinate spaces.

    Each box is independently normalised to its own image dimensions before
    computing intersection. This is the correct comparison when ref and user
    images have different resolutions or aspect ratios.
    """
    ax1, ay1, ax2, ay2 = _norm_box(a, img_h_a, img_w_a)
    bx1, by1, bx2, by2 = _norm_box(b, img_h_b, img_w_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter

    if union < 1e-9:
        return 0.0

    return float(inter / union)


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _figure_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Filter to figure-type regions only."""
    return [r for r in regions if r.region_type == RegionType.figure]


def _project_box(
    bbox:   BoundingBox,
    H_inv:  np.ndarray,
    user_h: int,
    user_w: int,
) -> BoundingBox:
    """
    Project a ref-space BoundingBox into user-image space via H_inv.
    All four corners are projected; result is the axis-aligned bounding rect,
    clipped to user image bounds.
    """
    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    corners = np.float32([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ]).reshape(-1, 1, 2)

    warped = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
    xs = np.clip(warped[:, 0], 0, user_w)
    ys = np.clip(warped[:, 1], 0, user_h)

    return BoundingBox(
        x=int(xs.min()), y=int(ys.min()),
        w=max(1, int(xs.max() - xs.min())),
        h=max(1, int(ys.max() - ys.min())),
    )


# =============================================================================
#  ORCHESTRATOR
# =============================================================================

def run(
    ref_regions:  list[LayoutRegion],
    user_regions: list[LayoutRegion],
    ref_h:  int,
    ref_w:  int,
    user_h: int,
    user_w: int,
    H_inv:  list[float] | None,
) -> tuple[list[TamperBox], list[TamperBox]]:
    """
    Match ref figure regions to user figure regions by normalised IoU.

    Matching strategy:
      - Only RegionType.figure regions are considered.
      - Greedy highest-cross-IoU matching: for each ref figure, find the
        user figure with the highest normalised IoU. Each user figure may
        only be consumed once (greedy closest-first, like spatial_second_pass).
      - IoU >= LAYOUT_IOU_THRESHOLD → matched → accepted (no flag).
      - IoU <  threshold            → ref figure unmatched → TamperBox in ref space.
      - Unmatched user figures      → TamperBox in user space.

    Unmatched ref boxes are additionally projected to user coordinate space
    via H_inv so the frontend can overlay "this region is missing" on the
    user image as well as the reference image.

    Args:
        ref_regions:  LayoutRegion list from reference SemanticMap.
        user_regions: LayoutRegion list from user SemanticMap (user pixel space).
        ref_h/w:      Reference image pixel dimensions.
        user_h/w:     User image pixel dimensions.
        H_inv:        Flat 9-element inverse homography (ref->user space).
                      Used to project unmatched ref boxes to user overlay.
                      May be None when alignment failed.

    Returns:
        (tamper_boxes_ref, tamper_boxes_user)
        Each TamperBox has source=TamperSource.layout and score=best_iou.
    """
    ref_figs  = _figure_regions(ref_regions)
    user_figs = _figure_regions(user_regions)

    log.info(
        "Layout: %d ref figures, %d user figures.",
        len(ref_figs), len(user_figs),
    )

    if not ref_figs and not user_figs:
        return [], []

    H_inv_mat: np.ndarray | None = None
    if H_inv is not None:
        H_inv_mat = np.array(H_inv, dtype=np.float64).reshape(3, 3)

    # ── Greedy IoU matching ──────────────────────────────────────────────────
    # Build all (iou, ref_idx, user_idx) candidates above zero
    candidates: list[tuple[float, int, int]] = []

    for ri, rr in enumerate(ref_figs):
        for ui, ur in enumerate(user_figs):
            score = _cross_iou(
                rr.bbox, ref_h, ref_w,
                ur.bbox, user_h, user_w,
            )
            if score > 0.0:
                candidates.append((score, ri, ui))

    # Sort descending by IoU so the best pairs are consumed first
    candidates.sort(key=lambda t: t[0], reverse=True)

    matched_ref:  set[int] = set()
    matched_user: set[int] = set()

    for score, ri, ui in candidates:
        if ri in matched_ref or ui in matched_user:
            continue
        if score >= LAYOUT_IOU_THRESHOLD:
            matched_ref.add(ri)
            matched_user.add(ui)
            log.debug(
                "Layout match: ref[%d] <-> user[%d]  IoU=%.3f",
                ri, ui, score,
            )
        # If score < threshold we do NOT consume the pair — both remain
        # available for potentially better matches at lower IoU.
        # In practice this only matters when multiple ref figures overlap
        # the same user figure region.

    # ── Unmatched ref figures → TamperBox in ref space ───────────────────────
    tamper_ref:  list[TamperBox] = []
    tamper_user: list[TamperBox] = []

    for ri, rr in enumerate(ref_figs):
        if ri in matched_ref:
            continue

        # Best IoU this ref figure achieved against any user figure
        best_iou = max(
            (_cross_iou(rr.bbox, ref_h, ref_w, ur.bbox, user_h, user_w)
             for ur in user_figs),
            default=0.0,
        )

        log.info(
            "Layout: unmatched ref figure [%d] bbox=%s  best_iou=%.3f",
            ri, rr.bbox, best_iou,
        )

        tamper_ref.append(TamperBox(
            bbox=rr.bbox,
            source=TamperSource.layout,
            score=round(best_iou, 4),
        ))

        # Also project to user space so the frontend can overlay both sides
        if H_inv_mat is not None:
            user_bbox = _project_box(rr.bbox, H_inv_mat, user_h, user_w)
            tamper_user.append(TamperBox(
                bbox=user_bbox,
                source=TamperSource.layout,
                score=round(best_iou, 4),
            ))

    # ── Unmatched user figures → TamperBox in user space ─────────────────────
    for ui, ur in enumerate(user_figs):
        if ui in matched_user:
            continue

        best_iou = max(
            (_cross_iou(rr.bbox, ref_h, ref_w, ur.bbox, user_h, user_w)
             for rr in ref_figs),
            default=0.0,
        )

        log.info(
            "Layout: unmatched user figure [%d] bbox=%s  best_iou=%.3f",
            ui, ur.bbox, best_iou,
        )

        tamper_user.append(TamperBox(
            bbox=ur.bbox,
            source=TamperSource.layout,
            score=round(best_iou, 4),
        ))

    log.info(
        "Layout complete: %d ref flags, %d user flags.",
        len(tamper_ref), len(tamper_user),
    )
    return tamper_ref, tamper_user