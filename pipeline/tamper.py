"""
pipeline/tamper.py — Masked visual tamper detection.

The "Teal Box" fix workflow:
  1. build_text_mask()   — draw black rectangles over all Cloud Vision text boxes
                           on the working image canvas, leaving only graphic regions.
  2. run_visual_diff()   — apply cell-level SSIM and pixel diff to unmasked regions
                           to detect altered graphics, logos, and background colours.

Design notes:
  - Text masking uses Cloud Vision bounding boxes (precise, semantic).
    This replaces the old keyword-heuristic nutrition panel mask and the
    FIX-B "only mask confirmed match polygons" approach — Cloud Vision gives
    us ALL text boxes upfront, including tampered ones, which is what we want
    to exclude from the visual diff.
  - SSIM is applied per cell (SSIM_CELL_SIZE × SSIM_CELL_SIZE blocks), not
    globally, to avoid false positives from global lighting differences between scans.
  - FIX-A (black rectangle detection) is preserved: near-black pixels receive
    full luminance weight in the diff map so solid black overlays are not absorbed
    into the background.
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


def build_text_mask(
    canvas_h: int,
    canvas_w: int,
    text_boxes: list[BoundingBox],
    pad: int = 6,
) -> np.ndarray:
    """
    Build a binary mask (uint8, 255 = text, 0 = graphic/background) from
    Cloud Vision text bounding boxes.

    Args:
        canvas_h:   Height of the working image.
        canvas_w:   Width of the working image.
        text_boxes: Word bounding boxes from SemanticMap (ref coordinate space).
        pad:        Extra pixels of padding around each box to absorb OCR margin.

    Returns:
        Mask array of shape (canvas_h, canvas_w), dtype uint8.
        255 = text region (to be excluded from visual diff).
        0   = graphic/background (to be diffed).
    """
    # TODO: implement
    # mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    # for box in text_boxes:
    #     x1 = max(0,        box.x - pad)
    #     y1 = max(0,        box.y - pad)
    #     x2 = min(canvas_w, box.x + box.w + pad)
    #     y2 = min(canvas_h, box.y + box.h + pad)
    #     mask[y1:y2, x1:x2] = 255
    # return mask
    raise NotImplementedError


def run_visual_diff(
    ref_work:     np.ndarray,
    user_aligned: np.ndarray,
    text_mask:    np.ndarray,
) -> list[BoundingBox]:
    """
    Detect visually tampered regions in graphic/background areas.

    Applies two complementary methods to the unmasked canvas:
      A) Cell-level pixel diff (fast, catches high-contrast overlays)
      B) Cell-level SSIM (structural similarity, catches subtle graphic swaps)

    Both are applied after zeroing masked (text) pixels so the comparison
    focuses exclusively on graphic content.

    FIX-A preserved: near-black pixels (L* < DARK_PIXEL_L_THRESH) receive
    full luminance weight so solid black rectangles are not absorbed into
    the background and remain detectable.

    Args:
        ref_work:     Reference working image (BGR, TAMPER_WORK_SIZE).
        user_aligned: User image warped to ref coordinate space (BGR, same size).
        text_mask:    Output of build_text_mask() — 255 = text region.

    Returns:
        List of BoundingBox tamper regions in working-image coordinates.
        Caller (main.py) scales back to full-resolution coordinates.
    """
    # TODO: implement
    raise NotImplementedError


def run(
    ref_img:      np.ndarray,
    user_img:     np.ndarray,
    H:            list[float] | None,
    H_inv:        list[float] | None,
    ref_semantic: object,   # SemanticMap — typed as object to avoid circular import
    inlier_ratio: float,
) -> tuple[list[TamperBox], list[TamperBox]]:
    """
    Orchestrate the full visual tamper detection pipeline.

    Workflow:
      1. Shrink both images to TAMPER_WORK_SIZE for speed.
      2. Warp user image using H (homography from align.py).
      3. Build text mask from ref SemanticMap word boxes.
      4. Run pixel diff + SSIM on unmasked regions.
      5. Scale detected boxes back to full-resolution coordinates.
      6. Project ref boxes to user coordinate space using H_inv.

    Args:
        ref_img:      Full-resolution reference image.
        user_img:     Full-resolution user scan.
        H:            Flat homography (user → ref) or None if alignment failed.
        H_inv:        Flat inverse homography or None.
        ref_semantic: SemanticMap for the reference image.
        inlier_ratio: Alignment quality from AlignmentResult.

    Returns:
        (tamper_boxes_ref, tamper_boxes_user)
        Each TamperBox has source=TamperSource.graphic or .chroma.
    """
    # TODO: implement
    raise NotImplementedError