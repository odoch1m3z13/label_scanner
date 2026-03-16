"""
Stage 4a – OCR-based text verification.

Uses PaddleOCR (primary) or pytesseract (fallback).
Compares text boxes between reference and aligned scan,
returning Defect objects for any text changes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import get_settings
from app.models.schemas import BoundingBox, ChangeType, Defect, Severity
from app.utils.geometry import iou

settings = get_settings()


@dataclass
class OcrBox:
    box: BoundingBox
    text: str
    confidence: float


# ── Engine wrappers ───────────────────────────────────────────────────────────

def _paddle_ocr(img: np.ndarray) -> list[OcrBox]:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    results = ocr.ocr(img, cls=True)
    boxes: list[OcrBox] = []
    if not results or results[0] is None:
        return boxes
    for line in results[0]:
        pts, (text, conf) = line
        pts = np.array(pts, dtype=np.int32)
        x, y, w, h = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max() - pts[:, 0].min()),
            int(pts[:, 1].max() - pts[:, 1].min()),
        )
        if conf >= settings.ocr_confidence_threshold:
            boxes.append(OcrBox(BoundingBox(x=x, y=y, w=max(1, w), h=max(1, h)), text, conf))
    return boxes


def _tesseract_ocr(img: np.ndarray) -> list[OcrBox]:
    import pytesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes: list[OcrBox] = []
    n = len(data["text"])
    for i in range(n):
        conf = int(data["conf"][i])
        text = data["text"][i].strip()
        if text and conf / 100.0 >= settings.ocr_confidence_threshold:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append(
                OcrBox(BoundingBox(x=x, y=y, w=max(1, w), h=max(1, h)), text, conf / 100.0)
            )
    return boxes


def run_ocr(img: np.ndarray) -> list[OcrBox]:
    try:
        return _paddle_ocr(img)
    except ImportError:
        return _tesseract_ocr(img)


# ── Serialisation helpers ─────────────────────────────────────────────────────

def boxes_to_dict(boxes: list[OcrBox]) -> list[dict[str, Any]]:
    return [
        {
            "box": {"x": b.box.x, "y": b.box.y, "w": b.box.w, "h": b.box.h},
            "text": b.text,
            "confidence": b.confidence,
        }
        for b in boxes
    ]


def dict_to_boxes(data: list[dict]) -> list[OcrBox]:
    return [
        OcrBox(
            BoundingBox(
                x=d["box"]["x"],
                y=d["box"]["y"],
                w=d["box"]["w"],
                h=d["box"]["h"],
            ),
            d["text"],
            d["confidence"],
        )
        for d in data
    ]


# ── Matching & diff ───────────────────────────────────────────────────────────

def _classify_change(ref_text: str, scan_text: str) -> Severity:
    """Heuristic: numeric/date changes are critical; minor typos are major."""
    ref_nums = re.findall(r"\d+[.,]?\d*", ref_text)
    scan_nums = re.findall(r"\d+[.,]?\d*", scan_text)
    if ref_nums != scan_nums:
        return Severity.CRITICAL
    if abs(len(ref_text) - len(scan_text)) > 5:
        return Severity.MAJOR
    return Severity.MINOR


def diff_ocr(
    ref_boxes: list[OcrBox],
    scan_boxes: list[OcrBox],
) -> tuple[list[Defect], float]:
    """
    Match reference OCR boxes to scan OCR boxes by spatial IoU,
    then diff text content.

    Returns (defects, duration_ms).
    """
    t0 = time.perf_counter()
    defects: list[Defect] = []

    used_scan: set[int] = set()

    for r in ref_boxes:
        best_iou = settings.ocr_iou_threshold
        best_idx = -1

        for i, s in enumerate(scan_boxes):
            if i in used_scan:
                continue
            score = iou(r.box, s.box)
            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_idx == -1:
            # Text region present in reference but missing in scan
            defects.append(
                Defect(
                    change_type=ChangeType.TEXT,
                    severity=Severity.CRITICAL,
                    description=f"Text region missing in scan: '{r.text}'",
                    ref_box=r.box,
                    scan_box=None,
                    ref_value=r.text,
                    scan_value=None,
                    confidence=0.95,
                )
            )
        else:
            used_scan.add(best_idx)
            s = scan_boxes[best_idx]
            # Normalise whitespace for comparison
            ref_norm = " ".join(r.text.split()).lower()
            scan_norm = " ".join(s.text.split()).lower()
            if ref_norm != scan_norm:
                severity = _classify_change(ref_norm, scan_norm)
                defects.append(
                    Defect(
                        change_type=ChangeType.TEXT,
                        severity=severity,
                        description=f"Text changed: '{r.text}' → '{s.text}'",
                        ref_box=r.box,
                        scan_box=s.box,
                        ref_value=r.text,
                        scan_value=s.text,
                        confidence=min(r.confidence, s.confidence),
                    )
                )

    # Text boxes in scan but not in reference → new/extra text
    for i, s in enumerate(scan_boxes):
        if i not in used_scan:
            defects.append(
                Defect(
                    change_type=ChangeType.TEXT,
                    severity=Severity.MAJOR,
                    description=f"Extra text in scan not in reference: '{s.text}'",
                    ref_box=None,
                    scan_box=s.box,
                    ref_value=None,
                    scan_value=s.text,
                    confidence=s.confidence,
                )
            )

    duration_ms = (time.perf_counter() - t0) * 1000
    return defects, duration_ms

def detect_text_color_changes(
    ref: np.ndarray,
    scan: np.ndarray,
    ref_boxes: list[OcrBox],
    scan_boxes: list[OcrBox],
) -> list[Defect]:
    """
    For each matched OCR box pair, sample the dominant text colour
    in the region and compare using ΔE. Flags colour-only text changes.
    """
    defects = []

    for r in ref_boxes:
        # Find matching scan box by IoU
        best_iou = 0.3
        best_s = None
        for s in scan_boxes:
            score = iou(r.box, s.box)
            if score > best_iou:
                best_iou = score
                best_s = s

        if best_s is None:
            continue

        # Only check if text content is the same (pure colour change)
        if r.text.strip().lower() != best_s.text.strip().lower():
            continue

        ref_color = _sample_text_color(ref, r.box)
        scan_color = _sample_text_color(scan, best_s.box)

        if ref_color is None or scan_color is None:
            continue

        # Compute LAB delta
        de = _lab_delta(ref_color, scan_color)

        if de > 15.0:  # noticeable colour shift
            defects.append(
                Defect(
                    change_type=ChangeType.COLOR,
                    severity=Severity.MAJOR,
                    description=f"Text colour changed: '{r.text}' (ΔE={de:.1f})",
                    ref_box=r.box,
                    scan_box=best_s.box,
                    confidence=min(1.0, de / 50.0),
                )
            )

    return defects


def _sample_text_color(img: np.ndarray, box: BoundingBox):
    """Sample the darkest (most ink-like) pixels in a text region."""
    x, y, w, h = box.to_xywh()
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Take the darkest 20% of pixels as the text colour
    threshold = np.percentile(gray, 20)
    mask = gray <= threshold
    if mask.sum() == 0:
        return None
    text_pixels = crop[mask].astype(np.float32)
    return text_pixels.mean(axis=0)  # BGR mean


def _lab_delta(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    """Simple LAB euclidean distance between two BGR colours."""
    def to_lab(bgr):
        px = np.uint8([[bgr]])
        return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
    lab1 = to_lab(bgr1)
    lab2 = to_lab(bgr2)
    return float(np.linalg.norm(lab1 - lab2))