"""
pipeline/barcode.py - Barcode region detection and HOG cosine comparison.

HOG (Histogram of Oriented Gradients) captures the vertical-bar gradient
structure of barcodes in a way that is invariant to JPEG quality, brightness,
and minor scale differences.

Why HOG for barcodes:
  - Barcodes have a fixed, well-understood gradient structure:
    alternating dark/light vertical bars produce a strong 90-degree
    orientation peak in the HOG histogram.
  - We resize every crop to a fixed landscape window (HOG_WIN_W x HOG_WIN_H)
    matching the natural barcode aspect ratio, avoiding the square-distortion
    problem that makes PHash/dHash unsuitable for arbitrary-aspect crops.
  - Cosine similarity is invariant to scale and brightness; it measures only
    the direction of the descriptor vector, not its magnitude.

Detection strategy (find_barcode_regions):
  Barcodes are characterised by high-density vertical edge columns.
  A standard 1D barcode has alternating dark/light vertical bars, so the
  x-direction Sobel gradient (dI/dx) has large values within the barcode
  column range and near-zero values outside it.

  Algorithm:
    1. Compute |dI/dx| (x-direction Sobel, absolute value).
    2. Sum edge magnitude per column, normalise by image height and 255.
    3. Mark columns with normalised density > BARCODE_EDGE_DENSITY as active.
    4. Run-length encode active columns into contiguous runs.
    5. Filter runs narrower than BARCODE_MIN_WIDTH_FRAC * image_width.
    6. Within each run, find the row extent by summing edge magnitude per row
       and keeping the contiguous band above 30% of the peak row energy.
    7. Filter runs shorter than BARCODE_MIN_HEIGHT_FRAC * image_height.

  Detection runs on the REFERENCE image only. We are checking whether the
  barcode at the known reference position has been altered — not locating
  barcodes in the user scan (which may be misaligned and confuse detection).

Comparison (run):
  For each detected barcode region:
    1. Crop from ref working image.
    2. Crop the same region from the aligned user working image.
    3. Compute HOG descriptors for both crops.
    4. If cosine_sim < BARCODE_HOG_THRESH, flag as barcode_changed.
    5. Scale boxes back to full-resolution ref coordinates.
    6. Project to user coordinate space via H_inv.

All functions are synchronous and CPU-bound. No I/O.
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

# Row energy threshold fraction: rows within a column run whose edge energy
# exceeds this fraction of the peak row energy are included in the barcode
# bounding box. 0.30 is empirically robust — it excludes the quiet zone
# margin above/below the bars while keeping the full bar height.
_ROW_ENERGY_FRAC = 0.30

# HOG descriptor shared instance — stateless after creation, safe to reuse.
_HOG = cv2.HOGDescriptor(
    _winSize    = (HOG_WIN_W, HOG_WIN_H),
    _blockSize  = (16, 16),
    _blockStride= (8, 8),
    _cellSize   = (8, 8),
    _nbins      = 9,
)


# =============================================================================
#  HOG UTILITIES  (already partially implemented in stub — kept as-is)
# =============================================================================

def hog_vec(gray_crop: np.ndarray) -> np.ndarray:
    """
    Compute a HOG descriptor for a greyscale crop.

    The crop is resized to (HOG_WIN_W, HOG_WIN_H) — a fixed landscape window
    that matches the natural barcode aspect ratio — before computation.
    Returns a zero vector for degenerate (empty) crops.
    """
    if gray_crop.size == 0:
        return np.zeros(1, np.float32)
    resized = cv2.resize(gray_crop, (HOG_WIN_W, HOG_WIN_H),
                         interpolation=cv2.INTER_AREA)
    return _HOG.compute(resized).flatten()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two HOG descriptor vectors in [0, 1].

    Returns 0.0 when either vector has near-zero norm (degenerate crop
    or solid uniform region).
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
#  BARCODE REGION DETECTION
# =============================================================================

def find_barcode_regions(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Locate barcode regions in a greyscale image using vertical-edge density.

    A barcode's dominant feature is a column of high x-direction edge density
    (alternating dark/light vertical bars). We scan column-by-column, group
    consecutive high-density columns into runs, then find the active row range
    within each run using per-row edge energy.

    Returns:
        List of (x, y, w, h) tuples in image pixel coordinates (working-image
        space). Empty list when no barcode-like regions are found.
    """
    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return []

    # Step 1: x-direction Sobel — detects vertical edges (barcode bar boundaries)
    sobel_x  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_mag = np.abs(sobel_x)   # shape (h, w), float32

    # Step 2: per-column edge density, normalised by height and max pixel value
    col_sum     = edge_mag.sum(axis=0)            # shape (w,)
    col_density = col_sum / max(1.0, h * 255.0)   # normalised [0, 1]

    # Step 3: mark active columns
    active_raw = (col_density > BARCODE_EDGE_DENSITY).astype(np.uint8)

    if not active_raw.any():
        return []

    # Step 4a: dilate active-column signal to bridge intra-barcode gaps.
    # A barcode's vertical bars produce alternating high/zero column densities
    # (bar edges are high, bar interiors are zero). Without dilation the
    # run-length encoder splits the barcode into dozens of 1-2 column runs,
    # all below the minimum width filter.
    # An 8-pixel kernel bridges up to 4 bar-width gaps at working resolution.
    kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
    active = cv2.dilate(active_raw.reshape(1, -1), kern).flatten().astype(bool)

    # Step 4b: run-length encode active columns into contiguous runs
    runs: list[tuple[int, int]] = []   # (x_start, x_end_exclusive)
    in_run   = False
    run_start = 0

    for x in range(w):
        if active[x] and not in_run:
            run_start = x
            in_run    = True
        elif not active[x] and in_run:
            runs.append((run_start, x))
            in_run = False
    if in_run:
        runs.append((run_start, w))

    # Step 5: filter runs narrower than minimum barcode width
    min_w_px = max(1, int(w * BARCODE_MIN_WIDTH_FRAC))
    runs     = [(x0, x1) for x0, x1 in runs if x1 - x0 >= min_w_px]

    # Step 6 & 7: find row extent for each column run, filter by min height
    min_h_px = max(1, int(h * BARCODE_MIN_HEIGHT_FRAC))
    regions: list[tuple[int, int, int, int]] = []

    for x0, x1 in runs:
        # Sum edge magnitude per row within this column band
        row_energy = edge_mag[:, x0:x1].sum(axis=1)   # shape (h,)
        peak_energy = float(row_energy.max())

        if peak_energy < 1.0:
            continue

        # Active rows: above _ROW_ENERGY_FRAC of peak
        active_rows = row_energy > (peak_energy * _ROW_ENERGY_FRAC)

        if not active_rows.any():
            continue

        # First and last active row (contiguous band)
        row_indices  = np.where(active_rows)[0]
        y0 = int(row_indices[0])
        y1 = int(row_indices[-1]) + 1   # exclusive

        bh = y1 - y0
        bw = x1 - x0

        if bh < min_h_px:
            continue

        regions.append((x0, y0, bw, bh))
        log.debug(
            "Barcode region found: x=%d y=%d w=%d h=%d  col_density=%.3f",
            x0, y0, bw, bh, col_density[x0:x1].mean(),
        )

    return regions


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _warp_user_to_work(
    user_img:  np.ndarray,
    H_mat:     np.ndarray,
    ref_scale: float,
    work_w:    int,
    work_h:    int,
) -> np.ndarray:
    """
    Warp the user image into ref working-image coordinate space.

    H_mat is in full-resolution pixel space. We scale it into working-image
    space using the same S_ref @ H @ S_user_inv pattern as tamper.py so that
    the coordinate systems are consistent between detectors.
    """
    user_h, user_w = user_img.shape[:2]
    user_scale     = min(1.0, TAMPER_WORK_SIZE / max(user_h, user_w))

    uw = max(1, int(user_w * user_scale))
    uh = max(1, int(user_h * user_scale))
    user_work = cv2.resize(user_img, (uw, uh), interpolation=cv2.INTER_AREA)

    S_ref      = np.diag([ref_scale,   ref_scale,   1.0])
    S_user_inv = np.diag([1/user_scale, 1/user_scale, 1.0])
    H_work     = S_ref @ H_mat @ S_user_inv

    return cv2.warpPerspective(
        user_work, H_work, (work_w, work_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _project_box_to_user(
    box:    tuple[int, int, int, int],
    H_inv:  np.ndarray,
    user_h: int,
    user_w: int,
) -> BoundingBox:
    """Project a full-resolution ref BoundingBox into user coordinate space."""
    x, y, bw, bh = box
    corners = np.float32([
        [x,      y      ],
        [x + bw, y      ],
        [x + bw, y + bh ],
        [x,      y + bh ],
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
    ref_img:  np.ndarray,
    user_img: np.ndarray,
    H:        list[float] | None,
    H_inv:    list[float] | None,
) -> tuple[list[TamperBox], list[TamperBox]]:
    """
    Detect barcode regions in the reference image and compare with the
    aligned user image using HOG cosine similarity.

    Regions where cosine_sim < BARCODE_HOG_THRESH are flagged as changed.

    Detection is performed on the REFERENCE image only. We are verifying that
    the barcode at the known reference position is intact — not searching for
    barcodes in an arbitrary user scan position.

    Workflow:
      1. Shrink ref to TAMPER_WORK_SIZE working canvas.
      2. Warp user into ref coordinate space at work size (if H available).
         If H is None, resize user to the same work dimensions (higher FP rate).
      3. Convert both working images to greyscale.
      4. Detect barcode regions in ref greyscale.
      5. For each region: compute HOG for ref crop and aligned user crop.
      6. If cosine_sim < BARCODE_HOG_THRESH, mark region as changed.
      7. Scale flagged work-size boxes back to full-resolution ref coordinates.
      8. Project full-res ref boxes into user coordinate space via H_inv.

    Args:
        ref_img:  Full-resolution reference image (BGR).
        user_img: Full-resolution user scan (BGR).
        H:        Flat 9-element homography (user->ref) or None.
        H_inv:    Flat 9-element inverse homography (ref->user) or None.

    Returns:
        (tamper_boxes_ref, tamper_boxes_user)
        Each TamperBox has source=TamperSource.barcode and score=cosine_sim.
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
        H_mat        = np.array(H, dtype=np.float64).reshape(3, 3)
        user_aligned = _warp_user_to_work(user_img, H_mat, ref_scale, work_w, work_h)
    else:
        user_aligned = cv2.resize(user_img, (work_w, work_h), interpolation=cv2.INTER_AREA)
        log.warning("Barcode: no homography — unaligned comparison (higher FP rate).")

    # Step 3: greyscale
    ref_gray  = cv2.cvtColor(ref_work,     cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_aligned, cv2.COLOR_BGR2GRAY)

    # Step 4: detect barcode regions in reference
    regions = find_barcode_regions(ref_gray)
    log.info("Barcode: %d candidate regions found in ref.", len(regions))

    if not regions:
        return [], []

    # Steps 5 & 6: HOG compare each region
    flagged_work: list[tuple[tuple[int, int, int, int], float]] = []

    for x, y, bw, bh in regions:
        ref_crop  = ref_gray [y:y+bh, x:x+bw]
        user_crop = user_gray[y:y+bh, x:x+bw]

        if ref_crop.size == 0 or user_crop.size == 0:
            continue

        ref_vec  = hog_vec(ref_crop)
        user_vec = hog_vec(user_crop)
        sim      = cosine_sim(ref_vec, user_vec)

        log.debug(
            "Barcode region x=%d y=%d w=%d h=%d  HOG_sim=%.3f  thresh=%.3f",
            x, y, bw, bh, sim, BARCODE_HOG_THRESH,
        )

        if sim < BARCODE_HOG_THRESH:
            flagged_work.append(((x, y, bw, bh), sim))
            log.info(
                "Barcode changed: region x=%d y=%d w=%d h=%d  sim=%.3f",
                x, y, bw, bh, sim,
            )

    if not flagged_work:
        return [], []

    # Step 7: scale flagged work-size boxes to full-resolution ref coordinates
    def _scale_to_full(box: tuple[int, int, int, int]) -> BoundingBox:
        x, y, bw, bh = box
        return BoundingBox(
            x=int(x / ref_scale), y=int(y / ref_scale),
            w=max(1, int(bw / ref_scale)), h=max(1, int(bh / ref_scale)),
        )

    tamper_ref: list[TamperBox] = []
    for work_box, sim in flagged_work:
        full_bbox = _scale_to_full(work_box)
        tamper_ref.append(TamperBox(
            bbox=full_bbox,
            source=TamperSource.barcode,
            score=round(sim, 4),
        ))

    # Step 8: project to user coordinate space
    tamper_user: list[TamperBox] = []
    if H_inv is not None:
        H_inv_mat = np.array(H_inv, dtype=np.float64).reshape(3, 3)
        for tb in tamper_ref:
            b     = tb.bbox
            u_box = _project_box_to_user((b.x, b.y, b.w, b.h), H_inv_mat, user_h, user_w)
            tamper_user.append(TamperBox(
                bbox=u_box,
                source=TamperSource.barcode,
                score=tb.score,
            ))

    log.info(
        "Barcode complete: %d ref boxes, %d user boxes.",
        len(tamper_ref), len(tamper_user),
    )
    return tamper_ref, tamper_user