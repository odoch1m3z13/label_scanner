from __future__ import annotations
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import get_settings

settings = get_settings()


@dataclass
class AlignmentResult:
    warped: np.ndarray
    homography: np.ndarray
    num_matches: int
    inlier_ratio: float
    method: str
    duration_ms: float


def _orb_match(ref_gray, scan_gray):
    orb = cv2.ORB_create(nfeatures=4000)

    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(scan_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("ORB failed")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        raise RuntimeError("Not enough matches")

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    return src, dst, len(good)


def align(ref: np.ndarray, scan: np.ndarray) -> AlignmentResult:
    t0 = time.perf_counter()

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    scan_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    method = "orb"
    src, dst, total = _orb_match(ref_gray, scan_gray)

    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        raise RuntimeError("Homography failed")

    inliers = int(mask.sum())
    ratio = inliers / max(total, 1)

    if ratio < 0.3:
        raise RuntimeError(f"Low alignment confidence: {ratio:.2f}")

    h, w = ref.shape[:2]
    warped = cv2.warpPerspective(scan, H, (w, h))

    return AlignmentResult(
        warped=warped,
        homography=H,
        num_matches=total,
        inlier_ratio=ratio,
        method=method,
        duration_ms=(time.perf_counter() - t0)*1000
    )