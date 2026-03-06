"""
pipeline/layout.py — Layout figure region comparison via IoU matching.

Compares figure/graphic regions between the reference SemanticMap and the
user SemanticMap (both produced by workers/cloud_vision.py).

Unmatched reference figures → something was removed or covered.
Unmatched user figures      → something was added or overlaid.
IoU-matched but pixel-different → graphic was swapped in place.
"""
from __future__ import annotations

import logging

import numpy as np

from config import LAYOUT_IOU_THRESHOLD
from models.schemas import BoundingBox, LayoutRegion, TamperBox, TamperSource

log = logging.getLogger(__name__)


def iou(a: BoundingBox, b: BoundingBox, img_h: int, img_w: int) -> float:
    """
    Compute IoU between two BoundingBoxes normalised to image dimensions.
    Normalisation makes the comparison scale-invariant (ref vs user may differ
    in absolute pixel dimensions if the photos were taken at different distances).
    """
    # TODO: implement
    raise NotImplementedError


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
    Match ref figure regions to user figure regions by IoU.

    Matching strategy:
      - For each ref figure, find the highest-IoU user figure.
      - If IoU ≥ LAYOUT_IOU_THRESHOLD → matched (accept).
      - If IoU < threshold → ref figure is unmatched → flag as tampered.
      - Unmatched user figures → flag as added.

    Args:
        ref_regions:  LayoutRegion list from reference SemanticMap.
        user_regions: LayoutRegion list from user SemanticMap.
        ref_h/w:      Reference image dimensions.
        user_h/w:     User image dimensions.
        H_inv:        Flat inverse homography for projecting ref boxes → user space.

    Returns:
        (tamper_boxes_ref, tamper_boxes_user)
        Each TamperBox has source=TamperSource.layout.
    """
    # TODO: implement
    raise NotImplementedError