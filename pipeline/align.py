"""
pipeline/align.py — ORB + RANSAC homography alignment and coordinate projection.

Two responsibilities:
  1. compute_homography(ref_img, user_img) → AlignmentResult
     Detects ORB keypoints in both images, matches with cross-check BFMatcher,
     and estimates the homography H that maps user → ref coordinate space.

  2. project_boxes(boxes, H_inv) → boxes
     Applies the inverse homography to Cloud Vision bounding boxes from the
     user image so they are expressed in reference image coordinate space.
     This avoids sending a perspective-warped image to Cloud Vision (which
     degrades OCR accuracy) — instead we warp the coordinates in post.

Design notes:
  - ORB is used (not SIFT) because Lowe's ratio test in SIFT discards repeated
    character keypoints on text-dense label surfaces, degrading the homography.
    ORB's cross-check + RANSAC geometric consensus is robust on these surfaces.
  - All functions are synchronous (CPU-bound, run in thread executor from main.py).
  - No FastAPI or storage imports — pure numpy/opencv logic.
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
from models.schemas import AlignmentResult, BoundingBox

log = logging.getLogger(__name__)


def _shrink(img: np.ndarray, max_px: int = TAMPER_WORK_SIZE) -> tuple[np.ndarray, float]:
    """Downscale img so its longest edge ≤ max_px. Returns (resized, scale)."""
    h, w = img.shape[:2]
    s    = min(1.0, max_px / max(h, w))
    if s == 1.0:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def compute_homography(
    ref_img:  np.ndarray,
    user_img: np.ndarray,
) -> AlignmentResult:
    """
    Estimate the homography H that maps user image coordinates → ref image
    coordinates using ORB keypoints and RANSAC.

    The homography is computed on downscaled working images for speed, then
    scaled back to full-resolution coordinates so it can be applied to the
    original-resolution images and bounding boxes.

    Returns:
        AlignmentResult with H, H_inv (both as flat 9-element row-major lists)
        and quality metrics. On failure, H and H_inv are None and status
        describes the failure mode.
    """
    # TODO: implement
    # ref_work, ref_s   = _shrink(ref_img)
    # user_work, user_s = _shrink(user_img)
    # ref_gray  = cv2.cvtColor(ref_work,  cv2.COLOR_BGR2GRAY)
    # user_gray = cv2.cvtColor(user_work, cv2.COLOR_BGR2GRAY)
    #
    # n = max(ORB_MIN_KEYPOINTS * 10, ORB_N_FEATURES)
    # orb = cv2.ORB_create(nfeatures=n)
    # kp1, d1 = orb.detectAndCompute(ref_gray,  None)
    # kp2, d2 = orb.detectAndCompute(user_gray, None)
    #
    # if d1 is None or d2 is None or len(kp1) < ORB_MIN_KEYPOINTS:
    #     return AlignmentResult(status="insufficient_keypoints")
    #
    # bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:200]
    #
    # if len(matches) < ORB_MIN_KEYPOINTS:
    #     return AlignmentResult(status="insufficient_matches")
    #
    # src_pts = np.float32([kp2[m.trainIdx].pt  for m in matches]).reshape(-1,1,2)
    # dst_pts = np.float32([kp1[m.queryIdx].pt  for m in matches]).reshape(-1,1,2)
    #
    # # Scale keypoint coordinates to full-resolution space before RANSAC
    # src_pts /= user_s;  dst_pts /= ref_s
    #
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    # inliers = int(mask.ravel().sum()) if mask is not None else 0
    #
    # if H is None or inliers < ORB_MIN_KEYPOINTS:
    #     return AlignmentResult(status="homography_failed", total_matches=len(matches))
    #
    # H_inv = np.linalg.inv(H)
    # return AlignmentResult(
    #     status="ok",
    #     inlier_ratio=inliers / max(1, len(matches)),
    #     inlier_count=inliers,
    #     total_matches=len(matches),
    #     H=H.flatten().tolist(),
    #     H_inv=H_inv.flatten().tolist(),
    # )
    raise NotImplementedError


def project_boxes(
    boxes: list[BoundingBox],
    H_inv: list[float],
    clip_w: int,
    clip_h: int,
) -> list[BoundingBox]:
    """
    Project a list of BoundingBoxes from user image coordinate space into
    reference image coordinate space using the inverse homography.

    This is the core of the "warp coordinates, not pixels" strategy:
      - Cloud Vision runs on the raw unwarped user image (best OCR quality).
      - The resulting bounding boxes are projected into ref space here.
      - The diff engine can then compare ref and user boxes directly.

    Args:
        boxes:  BoundingBoxes in user image pixel coordinates.
        H_inv:  Flat 9-element row-major 3×3 inverse homography matrix.
        clip_w: Reference image width  — clamps projected coordinates.
        clip_h: Reference image height — clamps projected coordinates.

    Returns:
        New list of BoundingBoxes in reference image coordinate space.
        Boxes that project outside [0, clip_w] × [0, clip_h] are clipped.
    """
    # TODO: implement
    # H = np.array(H_inv, dtype=np.float64).reshape(3, 3)
    # projected = []
    # for box in boxes:
    #     corners = np.float32([
    #         [box.x,         box.y        ],
    #         [box.x + box.w, box.y        ],
    #         [box.x + box.w, box.y + box.h],
    #         [box.x,         box.y + box.h],
    #     ]).reshape(-1, 1, 2)
    #     warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    #     xs = np.clip(warped[:, 0], 0, clip_w)
    #     ys = np.clip(warped[:, 1], 0, clip_h)
    #     projected.append(BoundingBox(
    #         x=int(xs.min()), y=int(ys.min()),
    #         w=int(xs.max() - xs.min()), h=int(ys.max() - ys.min()),
    #     ))
    # return projected
    raise NotImplementedError


def warp_image(
    user_img: np.ndarray,
    H:        list[float],
    ref_w:    int,
    ref_h:    int,
) -> np.ndarray:
    """
    Warp user_img into reference image coordinate space using homography H.
    Used by pipeline/tamper.py for pixel-level visual diff after alignment.

    Args:
        user_img: User scan as BGR numpy array.
        H:        Flat 9-element row-major homography matrix (user → ref).
        ref_w:    Output canvas width  (reference image width).
        ref_h:    Output canvas height (reference image height).

    Returns:
        Warped user image as BGR numpy array of shape (ref_h, ref_w, 3).
    """
    # TODO: implement
    # H_mat = np.array(H, dtype=np.float64).reshape(3, 3)
    # return cv2.warpPerspective(user_img, H_mat, (ref_w, ref_h))
    raise NotImplementedError