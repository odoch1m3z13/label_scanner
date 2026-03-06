"""
pipeline/barcode.py — Barcode region detection and HOG cosine comparison.

HOG (Histogram of Oriented Gradients) captures the vertical-bar gradient
structure of barcodes in a way that is invariant to JPEG quality, brightness,
and minor scale differences.

HOG is appropriate here because:
  - Barcodes have a fixed, well-understood gradient structure
    (alternating dark/light vertical bars → strong 90° orientation peak).
  - We resize to a fixed landscape window (HOG_WIN_W × HOG_WIN_H) that matches
    the natural barcode aspect ratio — no square-distortion problem.
  - Cosine similarity is scale/brightness invariant for the HOG descriptor vector.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import (
    BARCODE_EDGE_DENSITY,
    BARCODE_HOG_THRESH,
    BARCODE_MIN_HEIGHT_FRAC,
    BARCODE_MIN_WIDTH_FRAC,
    HOG_WIN_H,
    HOG_WIN_W,
    TAMPER_WORK_SIZE,
)
from models.schemas import BoundingBox, TamperBox, TamperSource

log = logging.getLogger(__name__)

# HOG descriptor — shared instance, stateless after creation.
_HOG = cv2.HOGDescriptor(
    _winSize    = (HOG_WIN_W, HOG_WIN_H),
    _blockSize  = (16, 16),
    _blockStride= (8, 8),
    _cellSize   = (8, 8),
    _nbins      = 9,
)


def hog_vec(gray_crop: np.ndarray) -> np.ndarray:
    """
    Compute a HOG descriptor for a grayscale crop.
    The crop is resized to HOG_WIN_W × HOG_WIN_H before computation.
    Returns a zero vector for degenerate (empty) crops.
    """
    if gray_crop.size == 0:
        return np.zeros(1, np.float32)
    resized = cv2.resize(gray_crop, (HOG_WIN_W, HOG_WIN_H),
                         interpolation=cv2.INTER_AREA)
    return _HOG.compute(resized).flatten()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two descriptor vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_barcode_regions(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Locate barcode regions in a grayscale image using vertical Sobel edge density.

    A barcode's dominant feature is its column of high vertical edge density.
    We scan column-by-column and group consecutive high-density columns into runs,
    then find the active row range within each run.

    Returns:
        List of (x, y, w, h) tuples in image pixel coordinates.
    """
    # TODO: implement
    raise NotImplementedError


def run(
    ref_img:  np.ndarray,
    user_img: np.ndarray,
    H:        list[float] | None,
    H_inv:    list[float] | None,
) -> tuple[list[TamperBox], list[TamperBox]]:
    """
    Detect barcode regions in the reference image and compare with the
    aligned user image using HOG cosine similarity.

    Regions where cosine_sim < BARCODE_HOG_THRESH are flagged as changed.

    Args:
        ref_img:  Full-resolution reference image (BGR).
        user_img: Full-resolution user scan (BGR).
        H:        Flat homography (user → ref) or None.
        H_inv:    Flat inverse homography or None.

    Returns:
        (tamper_boxes_ref, tamper_boxes_user)
        Each TamperBox has source=TamperSource.barcode and score=cosine_sim.
    """
    # TODO: implement
    raise NotImplementedError