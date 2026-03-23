from __future__ import annotations

import time
import cv2

from app.models.schemas import (
    ScanVerdict,
    ChangeType,
    Defect,
    Severity,
    BoundingBox,
)
from app.utils.geometry import annotate_image, iou


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

WEIGHTS = {
    ChangeType.TEXT: 10,
    ChangeType.BARCODE: 10,
    ChangeType.LOGO: 6,
    ChangeType.COLOR: 4,
    ChangeType.ANOMALY: 2,
}


# ────────────────────────────────────────────────
# BARCODE
# ────────────────────────────────────────────────

def _decode_barcodes(img):
    try:
        from pyzbar.pyzbar import decode
        results = decode(img)
        return [r.data.decode("utf-8", errors="replace") for r in results]
    except Exception:
        return []


def compare_barcodes(ref, scan, ref_barcodes):
    """
    Compare barcode values with fallback full-image boxes.
    Ensures visual highlighting even without exact localization.
    """
    t0 = time.perf_counter()
    defects = []

    scan_codes = _decode_barcodes(scan)

    ref_set = set(ref_barcodes or [])
    scan_set = set(scan_codes or [])

    h, w = ref.shape[:2]
    full_box = BoundingBox(x=0, y=0, w=w, h=h)

    # Missing
    for code in ref_set - scan_set:
        defects.append(
            Defect(
                change_type=ChangeType.BARCODE,
                severity=Severity.CRITICAL,
                description=f"Missing barcode: {code}",
                ref_value=code,
                scan_value=None,
                ref_box=full_box,
                scan_box=full_box,
                confidence=0.99,
            )
        )

    # Unexpected
    for code in scan_set - ref_set:
        defects.append(
            Defect(
                change_type=ChangeType.BARCODE,
                severity=Severity.CRITICAL,
                description=f"Unexpected barcode: {code}",
                ref_value=None,
                scan_value=code,
                ref_box=full_box,
                scan_box=full_box,
                confidence=0.99,
            )
        )

    duration = (time.perf_counter() - t0) * 1000
    return defects, duration


# ────────────────────────────────────────────────
# SAFETY: ENSURE BOXES
# ────────────────────────────────────────────────

def _ensure_boxes(defects, ref_shape):
    """
    Guarantee every defect has both ref_box and scan_box.
    Prevents missing annotations.
    """
    h, w = ref_shape[:2]
    full = BoundingBox(x=0, y=0, w=w, h=h)

    for d in defects:
        if d.ref_box is None and d.scan_box is None:
            d.ref_box = full
            d.scan_box = full

        elif d.ref_box is None:
            d.ref_box = d.scan_box

        elif d.scan_box is None:
            d.scan_box = d.ref_box

    return defects


# ────────────────────────────────────────────────
# DEDUPLICATION
# ────────────────────────────────────────────────

def _dedupe(defects):
    """
    Merge overlapping defects of same type.
    Prevents double counting from multiple stages.
    """
    out = []

    for d in defects:
        duplicate = False

        for o in out:
            if d.change_type != o.change_type:
                continue

            if d.scan_box and o.scan_box:
                if iou(d.scan_box, o.scan_box) > 0.5:
                    duplicate = True
                    break

        if not duplicate:
            out.append(d)

    return out


# ────────────────────────────────────────────────
# DECISION ENGINE
# ────────────────────────────────────────────────

def decide(ref, scan, stages, defects):
    """
    Final decision:
    - dedupe defects
    - ensure bounding boxes exist
    - weighted scoring
    - generate annotated outputs
    """

    # 🧠 CRITICAL: fix geometry consistency
    defects = _ensure_boxes(defects, ref.shape)

    # remove duplicates
    defects = _dedupe(defects)

    # scoring
    total_score = 0.0

    for d in defects:
        weight = WEIGHTS.get(d.change_type, 1)
        confidence = d.confidence if d.confidence is not None else 0.5
        total_score += weight * confidence

    # verdict
    if total_score > 15:
        verdict = ScanVerdict.FAIL
    elif total_score > 5:
        verdict = ScanVerdict.WARN
    else:
        verdict = ScanVerdict.PASS

    # annotate images
    annotated_ref = annotate_image(ref, defects, use_scan_box=False)
    annotated_scan = annotate_image(scan, defects, use_scan_box=True)

    return verdict, annotated_ref, annotated_scan