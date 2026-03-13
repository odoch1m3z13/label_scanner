"""
Stage 2 – Precision Alignment.

Primary:  LoFTR (deep feature matching) when torch is available.
Fallback: ORB + RANSAC homography.

Returns the scan image warped to match the reference frame,
plus alignment quality metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import get_settings

settings = get_settings()


@dataclass
class AlignmentResult:
    warped: np.ndarray              # scan warped into reference space
    homography: np.ndarray          # 3×3 homography matrix
    num_matches: int                # inlier match count
    method: str                     # "loftr" | "orb"
    duration_ms: float


# ── LoFTR (optional deep-matching) ───────────────────────────────────────────

def _try_loftr(ref_gray: np.ndarray, scan_gray: np.ndarray):
    """
    Attempt LoFTR matching via kornia.
    Returns (src_pts, dst_pts) or raises ImportError if kornia absent.
    """
    import torch
    from kornia.feature import LoFTR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained="outdoor").to(device).eval()

    def _prep(img: np.ndarray):
        t = torch.from_numpy(img).float()[None, None] / 255.0
        return t.to(device)

    data = {
        "image0": _prep(cv2.resize(ref_gray, (640, 480))),
        "image1": _prep(cv2.resize(scan_gray, (640, 480))),
    }

    with torch.no_grad():
        matcher(data)

    kp0 = data["keypoints0"].cpu().numpy()   # (N,2)
    kp1 = data["keypoints1"].cpu().numpy()

    # Scale back to original size
    sx = ref_gray.shape[1] / 640
    sy = ref_gray.shape[0] / 480
    kp0[:, 0] *= sx
    kp0[:, 1] *= sy

    sx = scan_gray.shape[1] / 640
    sy = scan_gray.shape[0] / 480
    kp1[:, 0] *= sx
    kp1[:, 1] *= sy

    return kp0, kp1


# ── ORB + RANSAC (robust fallback) ───────────────────────────────────────────

def _orb_match(ref_gray: np.ndarray, scan_gray: np.ndarray):
    orb = cv2.ORB_create(nfeatures=4000)

    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_scan, des_scan = orb.detectAndCompute(scan_gray, None)

    if des_ref is None or des_scan is None or len(kp_ref) < 4 or len(kp_scan) < 4:
        raise RuntimeError("Not enough ORB features detected")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des_ref, des_scan, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < settings.alignment_min_matches:
        raise RuntimeError(
            f"ORB: only {len(good)} good matches (need {settings.alignment_min_matches})"
        )

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scan[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return src_pts, dst_pts, good


# ── Public API ────────────────────────────────────────────────────────────────

def align(ref: np.ndarray, scan: np.ndarray) -> AlignmentResult:
    """
    Align *scan* to *ref*.

    1. Convert both to grayscale.
    2. Try LoFTR; fall back to ORB+RANSAC.
    3. Estimate homography with RANSAC.
    4. Warp scan into reference coordinate space.
    """
    t0 = time.perf_counter()

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    scan_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    method = "orb"
    num_matches = 0

    try:
        src_pts, dst_pts = _try_loftr(ref_gray, scan_gray)
        method = "loftr"
        num_matches = len(src_pts)
        H, mask = cv2.findHomography(
            dst_pts.reshape(-1, 1, 2),
            src_pts.reshape(-1, 1, 2),
            cv2.RANSAC,
            settings.alignment_ransac_threshold,
        )
    except Exception:
        src_pts, dst_pts, good = _orb_match(ref_gray, scan_gray)
        num_matches = len(good)
        H, mask = cv2.findHomography(
            dst_pts,
            src_pts,
            cv2.RANSAC,
            settings.alignment_ransac_threshold,
        )

    if H is None:
        raise RuntimeError("Homography estimation failed — images may be too different")

    h, w = ref.shape[:2]
    warped = cv2.warpPerspective(scan, H, (w, h), flags=cv2.INTER_LANCZOS4)

    duration_ms = (time.perf_counter() - t0) * 1000

    return AlignmentResult(
        warped=warped,
        homography=H,
        num_matches=int(num_matches),
        method=method,
        duration_ms=duration_ms,
    )