"""
Barcode verification + Decision Engine (Stage 4d + Stage 6).

Decision engine:
  – Aggregates all defects from all pipeline stages.
  – Applies critical-field rules.
  – Returns final ScanVerdict + annotated images.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.config import get_settings
from app.models.schemas import (
    BoundingBox,
    ChangeType,
    Defect,
    Severity,
    ScanVerdict,
    StageResult,
)
from app.utils.geometry import annotate_image

settings = get_settings()


# ── Barcode verification ──────────────────────────────────────────────────────

def _decode_barcodes(img: np.ndarray) -> list[str]:
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        results = pyzbar_decode(img)
        return [r.data.decode("utf-8", errors="replace") for r in results]
    except ImportError:
        pass
    return []


def compare_barcodes(
    ref: np.ndarray,
    scan: np.ndarray,
    ref_barcodes: list[str],
) -> tuple[list[Defect], float]:
    t0 = time.perf_counter()
    defects: list[Defect] = []

    scan_barcodes = _decode_barcodes(scan)

    ref_set = set(ref_barcodes)
    scan_set = set(scan_barcodes)

    for code in ref_set - scan_set:
        defects.append(
            Defect(
                change_type=ChangeType.BARCODE,
                severity=Severity.CRITICAL,
                description=f"Barcode missing in scan: {code}",
                ref_value=code,
                scan_value=None,
                confidence=0.99,
            )
        )

    for code in scan_set - ref_set:
        defects.append(
            Defect(
                change_type=ChangeType.BARCODE,
                severity=Severity.CRITICAL,
                description=f"Unexpected barcode in scan: {code}",
                ref_value=None,
                scan_value=code,
                confidence=0.99,
            )
        )

    duration_ms = (time.perf_counter() - t0) * 1000
    return defects, duration_ms


# ── Decision engine ───────────────────────────────────────────────────────────

def _has_critical_field_change(defect: Defect) -> bool:
    """
    Check whether a text defect touches a critical label field
    (product name, weight, expiry, batch, barcode).
    """
    if defect.change_type not in (ChangeType.TEXT, ChangeType.BARCODE):
        return False
    for field in settings.critical_fields:
        kw = field.replace("_", " ").lower()
        for val in (defect.ref_value or "", defect.scan_value or "", defect.description):
            if kw in val.lower():
                return True
    return False


def decide(
    ref: np.ndarray,
    scan_aligned: np.ndarray,
    stages: list[StageResult],
    all_defects: list[Defect],
) -> tuple[ScanVerdict, np.ndarray, np.ndarray]:
    """
    Final decision:
      1. Count critical-severity defects
      2. Check critical-field text changes
      3. Return verdict + annotated images

    Returns:
        verdict
        annotated_ref   – reference image with green boxes (ref defect locations)
        annotated_scan  – scan image with red/orange boxes (scan defect locations)
    """
    critical_count = sum(
        1 for d in all_defects
        if d.severity == Severity.CRITICAL or _has_critical_field_change(d)
    )
    major_count = sum(1 for d in all_defects if d.severity == Severity.MAJOR)

    if critical_count > 0:
        verdict = ScanVerdict.FAIL
    elif major_count > 2:
        verdict = ScanVerdict.WARN
    elif len(all_defects) > 0:
        verdict = ScanVerdict.WARN
    else:
        verdict = ScanVerdict.PASS

    # Annotate reference (green = expected location)
    annotated_ref = annotate_image(ref, all_defects, use_scan_box=False)
    # Overlay "REFERENCE" header
    _stamp(annotated_ref, "REFERENCE", (0, 180, 0))

    # Annotate scan (red/orange = found location)
    annotated_scan = annotate_image(scan_aligned, all_defects, use_scan_box=True)
    color = (0, 0, 200) if verdict == ScanVerdict.FAIL else (0, 140, 255)
    _stamp(annotated_scan, f"SCAN – {verdict.value}", color)

    return verdict, annotated_ref, annotated_scan


def _stamp(img: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 36), color, -1)
    cv2.putText(
        img, text, (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA,
    )