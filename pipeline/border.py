"""
pipeline/border.py — White border / padding delta detection.

Detects when the user scan has significantly more white padding around the
label content than the reference image — a sign of image compositing or
re-saving with added white space.

Checks all four edges independently. Flags only borders that are:
  - Larger than MIN_BORDER_DELTA_PX extra pixels vs reference.
  - At least MIN_BORDER_FRAC of the image dimension.
  - Sufficiently uniform white (BORDER_UNIFORMITY_FRAC).
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


def content_bounds(img_bgr: np.ndarray) -> BoundingBox | None:
    """
    Find the bounding box of non-white content in the image.
    Operates on a thumbnail for speed (BOUNDS_THUMB_PX).

    Returns None if the entire image is white.
    """
    # TODO: implement
    raise NotImplementedError


def measure_border_uniformity(
    img_bgr:   np.ndarray,
    side:      str,
    thickness: int,
) -> float:
    """
    Return the fraction of pixels in a border strip that are white.
    side: "left" | "right" | "top" | "bottom"
    """
    # TODO: implement
    raise NotImplementedError


def run(
    ref_img:  np.ndarray,
    user_img: np.ndarray,
) -> list[TamperBox]:
    """
    Compare white border widths between ref and user on all four edges.

    Returns a list of TamperBox (in user image coordinates) for each
    edge where the user image has significantly more padding.
    Each TamperBox has source=TamperSource.white_border.
    """
    # TODO: implement
    raise NotImplementedError