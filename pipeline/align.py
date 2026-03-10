"""
pipeline/align.py — ORB + RANSAC homography alignment and coordinate projection.

Three responsibilities:
  1. compute_homography(ref_img, user_img) → AlignmentResult
     Detects ORB keypoints in both images, matches with cross-check BFMatcher,
     and estimates the homography H that maps user → ref coordinate space.

  2. project_boxes / project_words
     Applies the inverse homography to Cloud Vision bounding boxes and word
     polygons from the user image so they are expressed in reference image
     coordinate space.
     This avoids sending a perspective-warped image to Cloud Vision (which
     degrades OCR accuracy on rotated text) — instead we warp the coordinates
     in post, keeping Cloud Vision's input as clean as possible.

  3. warp_image(user_img, H, ref_w, ref_h) → np.ndarray
     Warps the user image into reference coordinate space for pixel-level
     visual diff in pipeline/tamper.py.

Why ORB, not SIFT:
  SIFT uses Lowe's ratio test to filter ambiguous matches. On text-dense label
  surfaces there are dozens of visually identical characters (e.g. lowercase 'e',
  'a', 'o'). Lowe's ratio test discards almost all of them because no single
  match is significantly better than its second-best neighbour. The result is
  too few good matches for a reliable homography.
  ORB uses cross-check matching (each descriptor's best match must be mutual)
  followed by RANSAC geometric consensus — robust on repeating structures
  because RANSAC filters outliers by spatial consistency, not local uniqueness.

Preprocessing for ORB:
  CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the
  grayscale working image before ORB detection. This improves keypoint detection
  on low-contrast regions (light-coloured label backgrounds, faded print).
  It is applied only to the working (downscaled) image and has no effect on
  the coordinates returned — they are always in full-resolution pixel space.

All functions are synchronous (CPU-bound, run in thread executor from main.py).
No FastAPI or storage imports — pure numpy/opencv logic.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from config import (
    ORB_MIN_KEYPOINTS,
    ORB_N_FEATURES,
    RANSAC_REPROJ_THRESHOLD,
    TAMPER_WORK_SIZE,
)
from models.schemas import AlignmentResult, BoundingBox, Polygon, WordEntry

log = logging.getLogger(__name__)

# Maximum number of cross-checked matches passed to RANSAC.
# Keeping the 200 best (lowest Hamming distance) reduces noise without
# losing the geometric coverage RANSAC needs.
_MAX_MATCHES = 200

# CLAHE parameters for ORB preprocessing.
_CLAHE_CLIP  = 3.0
_CLAHE_TILE  = (8, 8)


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _shrink(img: np.ndarray, max_px: int = TAMPER_WORK_SIZE) -> tuple[np.ndarray, float]:
    """
    Downscale img so its longest edge ≤ max_px.
    Returns (resized_image, scale_factor).
    scale_factor < 1.0 means the image was shrunk.
    """
    h, w = img.shape[:2]
    s    = min(1.0, max_px / max(h, w))
    if s == 1.0:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def _preprocess_for_orb(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to a grayscale image to improve ORB keypoint detection on
    low-contrast label areas (light backgrounds, subtle texture differences).
    """
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE)
    return clahe.apply(gray)


def _h_matrix(flat: list[float]) -> np.ndarray:
    """Reshape a flat 9-element list into a 3×3 float64 numpy matrix."""
    return np.array(flat, dtype=np.float64).reshape(3, 3)


def _project_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a 3×3 homography to an array of 2D points.

    Args:
        pts: Shape (N, 2) float32 array of (x, y) points.
        H:   3×3 homography matrix.

    Returns:
        Shape (N, 2) float32 array of projected (x, y) points.
    """
    reshaped  = pts.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(reshaped, H)
    return projected.reshape(-1, 2)


# =============================================================================
#  PUBLIC FUNCTIONS
# =============================================================================

def compute_homography(
    ref_img:  np.ndarray,
    user_img: np.ndarray,
) -> AlignmentResult:
    """
    Estimate the homography H that maps user image coordinates → ref image
    coordinates using ORB keypoints and RANSAC.

    Pipeline:
      1. Shrink both images to TAMPER_WORK_SIZE for fast keypoint detection.
      2. Apply CLAHE to improve detection on low-contrast label backgrounds.
      3. Detect ORB keypoints and compute binary descriptors.
      4. Cross-check BFMatcher — keeps only mutually best matches.
      5. Take the top _MAX_MATCHES by Hamming distance.
      6. Scale keypoint coordinates back to full-resolution pixel space.
         (Critical: homography must be in full-res space so it applies
          correctly to full-res images and Cloud Vision bounding boxes.)
      7. RANSAC findHomography — filters geometric outliers.
      8. Invert H to get H_inv (ref → user, used for projecting ref boxes
         into user coordinate space for the frontend tamper overlay).

    Returns:
        AlignmentResult. On any failure, H and H_inv are None and `status`
        describes the failure mode. The caller (main.py) must check
        result.status == "ok" before using H / H_inv.
    """
    ref_h,  ref_w  = ref_img.shape[:2]
    user_h, user_w = user_img.shape[:2]

    # ── Step 1: Downscale to working size ─────────────────────────────────────
    ref_work,  ref_s  = _shrink(ref_img)
    user_work, user_s = _shrink(user_img)

    # ── Step 2: CLAHE → greyscale ─────────────────────────────────────────────
    ref_gray  = _preprocess_for_orb(cv2.cvtColor(ref_work,  cv2.COLOR_BGR2GRAY))
    user_gray = _preprocess_for_orb(cv2.cvtColor(user_work, cv2.COLOR_BGR2GRAY))

    # ── Step 3: ORB keypoint detection ────────────────────────────────────────
    n_features = max(ORB_MIN_KEYPOINTS * 10, ORB_N_FEATURES)
    orb        = cv2.ORB_create(nfeatures=n_features)
    kp1, d1    = orb.detectAndCompute(ref_gray,  None)
    kp2, d2    = orb.detectAndCompute(user_gray, None)

    log.info(
        "ORB: ref=%d kp  user=%d kp  (working size %dx%d / %dx%d)",
        len(kp1), len(kp2),
        ref_work.shape[1],  ref_work.shape[0],
        user_work.shape[1], user_work.shape[0],
    )

    if d1 is None or d2 is None or len(kp1) < ORB_MIN_KEYPOINTS or len(kp2) < ORB_MIN_KEYPOINTS:
        log.warning("ORB: insufficient keypoints (ref=%d, user=%d, min=%d).",
                    len(kp1), len(kp2), ORB_MIN_KEYPOINTS)
        return AlignmentResult(status="insufficient_keypoints")

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:_MAX_MATCHES]

    log.info("ORB: %d cross-checked matches (cap %d).", len(matches), _MAX_MATCHES)

    if len(matches) < ORB_MIN_KEYPOINTS:
        log.warning("ORB: insufficient matches (%d < %d).", len(matches), ORB_MIN_KEYPOINTS)
        return AlignmentResult(status="insufficient_matches", total_matches=len(matches))

    src_pts = np.float32([kp2[m.trainIdx].pt  for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt  for m in matches]).reshape(-1, 1, 2)
    src_pts = src_pts / user_s
    dst_pts = dst_pts / ref_s

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

    inliers = int(mask.ravel().sum()) if mask is not None else 0
    log.info("RANSAC: %d inliers / %d matches  (ratio=%.2f)",
             inliers, len(matches), inliers / max(1, len(matches)))

    if H is None or inliers < ORB_MIN_KEYPOINTS:
        log.warning("Homography failed (inliers=%d, min=%d).", inliers, ORB_MIN_KEYPOINTS)
        return AlignmentResult(
            status="homography_failed",
            total_matches=len(matches),
            inlier_count=inliers,
        )

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        log.warning("Homography matrix is singular — cannot invert.")
        return AlignmentResult(
            status="singular_homography",
            total_matches=len(matches),
            inlier_count=inliers,
        )

    inlier_ratio = inliers / max(1, len(matches))
    log.info("Alignment OK — inlier_ratio=%.3f", inlier_ratio)

    return AlignmentResult(
        status="ok",
        inlier_ratio=round(inlier_ratio, 4),
        inlier_count=inliers,
        total_matches=len(matches),
        H=H.flatten().tolist(),
        H_inv=H_inv.flatten().tolist(),
    )


def project_boxes(
    boxes:  list[BoundingBox],
    H_inv:  list[float],
    clip_w: int,
    clip_h: int,
) -> list[BoundingBox]:
    """
    Project BoundingBoxes from user image coordinate space into reference
    image coordinate space using the inverse homography.

    Each box's four corners are projected through H_inv individually, then
    a new axis-aligned bounding box is computed from the projected corners.
    This correctly handles rotation and shear in the homography — a box that
    was rotated in the user image becomes a larger axis-aligned box in ref
    space that tightly contains all four projected corners.

    Args:
        boxes:  BoundingBoxes in user image pixel coordinates.
        H_inv:  Flat 9-element row-major 3×3 inverse homography (user → ref).
        clip_w: Reference image width  — projected coordinates are clamped.
        clip_h: Reference image height — projected coordinates are clamped.

    Returns:
        New list of BoundingBoxes in reference image pixel coordinates.
        Maintains 1:1 correspondence with the input list.
    """
    if not boxes:
        return []

    H = _h_matrix(H_inv)
    projected: list[BoundingBox] = []

    for box in boxes:
        corners = np.float32([
            [box.x,           box.y          ],
            [box.x + box.w,   box.y          ],
            [box.x + box.w,   box.y + box.h  ],
            [box.x,           box.y + box.h  ],
        ])
        warped = _project_points(corners, H)

        xs = np.clip(warped[:, 0], 0, clip_w)
        ys = np.clip(warped[:, 1], 0, clip_h)

        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())

        projected.append(BoundingBox(
            x=x0, y=y0,
            w=max(1, x1 - x0),
            h=max(1, y1 - y0),
        ))

    return projected


def project_words(
    words:  list[WordEntry],
    H_inv:  list[float],
    clip_w: int,
    clip_h: int,
) -> list[WordEntry]:
    """
    Project all spatial fields of a WordEntry list (bbox and polygon) from
    user image coordinate space into reference image coordinate space.

    Both bbox and polygon are updated so that downstream consumers (the colour
    sampler in pipeline/colour.py and the text mask builder in pipeline/tamper.py)
    work in a consistent reference coordinate space.

    Returns new WordEntry objects — the originals are not mutated.

    Args:
        words:  WordEntry list from Cloud Vision user scan (user coord space).
        H_inv:  Flat 9-element inverse homography matrix (user → ref coords).
        clip_w: Reference image width.
        clip_h: Reference image height.

    Returns:
        New list of WordEntry with bbox and polygon in reference coord space.
        block_id, para_id, text, confidence are copied unchanged.
    """
    if not words:
        return []

    H = _h_matrix(H_inv)
    result: list[WordEntry] = []

    for word in words:
        poly_pts = np.float32(word.polygon.points)       
        warped_poly = _project_points(poly_pts, H)
        warped_poly[:, 0] = np.clip(warped_poly[:, 0], 0, clip_w)
        warped_poly[:, 1] = np.clip(warped_poly[:, 1], 0, clip_h)
        new_polygon = Polygon(points=[
            (int(p[0]), int(p[1])) for p in warped_poly
        ])

        # ── Derive bbox from projected polygon (tightest axis-aligned rect) ───
        xs = warped_poly[:, 0]
        ys = warped_poly[:, 1]
        new_bbox = BoundingBox(
            x=int(xs.min()),
            y=int(ys.min()),
            w=max(1, int(xs.max() - xs.min())),
            h=max(1, int(ys.max() - ys.min())),
        )

        result.append(WordEntry(
            text=word.text,
            confidence=word.confidence,
            bbox=new_bbox,
            polygon=new_polygon,
            block_id=word.block_id,
            para_id=word.para_id,
        ))

    return result


def warp_image(
    user_img: np.ndarray,
    H:        list[float],
    ref_w:    int,
    ref_h:    int,
) -> np.ndarray:
    """
    Warp user_img into reference image coordinate space using homography H.
    Used by pipeline/tamper.py for pixel-level visual diff after alignment.

    The warped image has the same canvas size as the reference image
    (ref_w × ref_h). Pixels that fall outside the user image after warping
    are filled with black (0, 0, 0) — these are excluded from the visual
    diff by the valid-pixel mask in tamper.py.

    Args:
        user_img: User scan as BGR numpy array (any size).
        H:        Flat 9-element row-major homography matrix (user → ref).
        ref_w:    Output canvas width  (reference image width in pixels).
        ref_h:    Output canvas height (reference image height in pixels).

    Returns:
        Warped user image as BGR numpy array of shape (ref_h, ref_w, 3).
        Black pixels indicate areas outside the user image boundary.
    """
    H_mat = _h_matrix(H)
    return cv2.warpPerspective(
        user_img, H_mat, (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )