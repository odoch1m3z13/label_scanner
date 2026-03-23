from __future__ import annotations
import time
import numpy as np
from rapidfuzz import fuzz

from app.config import get_settings
from app.models.schemas import BoundingBox, Defect, ChangeType, Severity
from app.utils.geometry import iou

settings = get_settings()

_ocr_model = None


def _get_ocr():
    global _ocr_model
    if _ocr_model is None:
        from paddleocr import PaddleOCR
        _ocr_model = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_model


class OcrBox:
    def __init__(self, box, text, confidence):
        self.box = box
        self.text = text
        self.confidence = confidence


def run_ocr(img: np.ndarray):
    ocr = _get_ocr()
    results = ocr.ocr(img, cls=True)

    boxes = []
    if not results or results[0] is None:
        return boxes

    for line in results[0]:
        pts, (text, conf) = line
        if conf < settings.ocr_confidence_threshold:
            continue

        pts = np.array(pts)
        x, y = int(pts[:,0].min()), int(pts[:,1].min())
        w, h = int(pts[:,0].max() - x), int(pts[:,1].max() - y)

        boxes.append(
            OcrBox(
                BoundingBox(x=x, y=y, w=w, h=h),
                text,
                conf
            )
        )

    return boxes


def _sim(a, b):
    return fuzz.ratio(a, b) / 100


def diff_ocr(ref_boxes, scan_boxes):
    t0 = time.perf_counter()
    defects = []
    used = set()

    for r in ref_boxes:
        best_i = -1
        best_score = 0

        for i, s in enumerate(scan_boxes):
            if i in used:
                continue

            score = 0.6 * iou(r.box, s.box) + 0.4 * _sim(r.text, s.text)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i == -1 or best_score < 0.5:
            defects.append(Defect(
                change_type=ChangeType.TEXT,
                severity=Severity.CRITICAL,
                description=f"Missing text: {r.text}",
                ref_box=r.box,
                scan_box=None,
                confidence=0.9
            ))
            continue

        used.add(best_i)
        s = scan_boxes[best_i]

        sim = _sim(r.text, s.text)

        if sim < 0.9:
            severity = Severity.CRITICAL if any(c.isdigit() for c in r.text) else Severity.MAJOR
            defects.append(Defect(
                change_type=ChangeType.TEXT,
                severity=severity,
                description=f"{r.text} → {s.text}",
                ref_box=r.box,
                scan_box=s.box,
                confidence=sim
            ))

    duration = (time.perf_counter() - t0)*1000
    return defects, duration

def dict_to_boxes(data):
    """
    Convert stored OCR JSON into OcrBox objects
    """
    boxes = []

    if not data:
        return boxes

    for item in data:
        try:
            box = BoundingBox(
                x=item["box"]["x"],
                y=item["box"]["y"],
                w=item["box"]["w"],
                h=item["box"]["h"],
            )

            boxes.append(OcrBox(
                box=box,
                text=item.get("text", ""),
                confidence=item.get("confidence", 1.0),
            ))
        except Exception:
            continue

    return boxes