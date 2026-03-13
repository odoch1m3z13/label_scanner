"""
Stage 4c – Color Verification.

Computes CIEDE2000 (ΔE) color difference between reference and scan
across a grid of sample regions. Flags regions exceeding the threshold.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.config import get_settings
from app.models.schemas import BoundingBox, ChangeType, Defect, Severity
from app.utils.geometry import mask_to_boxes

settings = get_settings()


# ── ΔE CIEDE2000 ─────────────────────────────────────────────────────────────

def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Vectorised CIEDE2000 for two LAB colour values (shape: (3,)).
    Returns scalar ΔE.
    """
    # Unpack
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    # Step 1: a' adjustment
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_avg ** 7 / (C_avg ** 7 + 25 ** 7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    # Step 2: differences
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0
    elif abs(dhp) > 180:
        dhp = dhp - 360 if dhp > 0 else dhp + 360

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    # Step 3: CIEDE2000 weighting
    Lp_avg = (L1 + L2) / 2
    Cp_avg = (C1p + C2p) / 2

    if C1p * C2p == 0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_avg = (h1p + h2p) / 2
    else:
        hp_avg = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2

    T = (
        1
        - 0.17 * np.cos(np.radians(hp_avg - 30))
        + 0.24 * np.cos(np.radians(2 * hp_avg))
        + 0.32 * np.cos(np.radians(3 * hp_avg + 6))
        - 0.20 * np.cos(np.radians(4 * hp_avg - 63))
    )

    SL = 1 + 0.015 * (Lp_avg - 50) ** 2 / np.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    d_theta = 30 * np.exp(-(((hp_avg - 275) / 25) ** 2))
    RC = 2 * np.sqrt(Cp_avg ** 7 / (Cp_avg ** 7 + 25 ** 7))
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    dE = np.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return float(dE)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_color_profile(img: np.ndarray, grid: int = 8) -> list[dict[str, Any]]:
    """
    Sample dominant colours on a grid. Used when registering the reference.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cell_h, cell_w = h // grid, w // grid
    profile: list[dict[str, Any]] = []
    for row in range(grid):
        for col in range(grid):
            y0, x0 = row * cell_h, col * cell_w
            cell = lab[y0: y0 + cell_h, x0: x0 + cell_w]
            mean_lab = cell.reshape(-1, 3).mean(axis=0).tolist()
            profile.append(
                {
                    "row": row,
                    "col": col,
                    "x": x0,
                    "y": y0,
                    "w": cell_w,
                    "h": cell_h,
                    "lab": mean_lab,
                }
            )
    return profile


def compare_colors(
    ref: np.ndarray,
    scan: np.ndarray,
    ref_profile: list[dict[str, Any]],
) -> tuple[list[Defect], float]:
    """
    Compare scan colour grid against stored reference profile.
    Returns (defects, duration_ms).
    """
    t0 = time.perf_counter()
    defects: list[Defect] = []

    scan_lab = cv2.cvtColor(scan, cv2.COLOR_BGR2LAB).astype(np.float32)

    for cell in ref_profile:
        x, y, cw, ch = cell["x"], cell["y"], cell["w"], cell["h"]
        scan_cell = scan_lab[y: y + ch, x: x + cw]
        if scan_cell.size == 0:
            continue
        scan_mean = scan_cell.reshape(-1, 3).mean(axis=0)
        ref_mean = np.array(cell["lab"], dtype=np.float32)

        de = _delta_e_2000(ref_mean, scan_mean)

        if de > settings.color_delta_e_threshold:
            severity = Severity.CRITICAL if de > settings.color_delta_e_threshold * 3 else Severity.MAJOR
            box = BoundingBox(x=x, y=y, w=max(1, cw), h=max(1, ch))
            defects.append(
                Defect(
                    change_type=ChangeType.COLOR,
                    severity=severity,
                    description=f"Color shift ΔE={de:.1f} at grid ({cell['row']},{cell['col']})",
                    ref_box=box,
                    scan_box=box,
                    confidence=min(1.0, de / (settings.color_delta_e_threshold * 5)),
                )
            )

    duration_ms = (time.perf_counter() - t0) * 1000
    return defects, duration_ms