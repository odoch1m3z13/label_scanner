import cv2
import numpy as np
from rapidfuzz import fuzz

from app.models.schemas import Defect, ChangeType, Severity, BoundingBox
from app.pipeline.ocr import run_ocr


def check_template_text(ref, scan, template):
    defects = []

    for region in template:
        if region["type"] != "text":
            continue

        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        scan_crop = scan[y:y+h, x:x+w]

        boxes = run_ocr(scan_crop)
        scan_text = " ".join([b.text for b in boxes]).strip()

        expected = region.get("expected_text", "").strip()

        if region.get("type") == "expiry":
            match = scan_text == expected
        elif region.get("type") == "numeric":
            match = scan_text == expected
        elif region.get("strict", False):
            match = scan_text == expected
        else:
            match = fuzz.ratio(scan_text, expected) > 92

        if not match:
            defects.append(
                Defect(
                    change_type=ChangeType.TEXT,
                    severity=Severity.CRITICAL if region.get("strict") else Severity.MAJOR,
                    description=f"{expected} → {scan_text}",
                    ref_box=BoundingBox(x=x, y=y, w=w, h=h),
                    scan_box=BoundingBox(x=x, y=y, w=w, h=h),
                    confidence=0.9,
                )
            )

    return defects

def build_template_mask(shape, template):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for r in template:
        x, y, w_, h_ = r["x"], r["y"], r["w"], r["h"]
        mask[y:y+h_, x:x+w_] = 255

    return mask

def detect_unexpected_changes(ref, scan, template):
    defects = []

    mask = build_template_mask(ref.shape, template)

    # compute difference
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(g1, g2)

    # ONLY look outside template
    outside = cv2.bitwise_and(diff, diff, mask=cv2.bitwise_not(mask))

    # threshold
    _, thresh = cv2.threshold(outside, 25, 255, cv2.THRESH_BINARY)

    # clean noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find blobs
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w * h < 500:  # ignore tiny noise
            continue

        defects.append(
            Defect(
                change_type=ChangeType.ANOMALY,
                severity=Severity.MAJOR,
                description="Unexpected content detected",
                ref_box=BoundingBox(x=x, y=y, w=w, h=h),
                scan_box=BoundingBox(x=x, y=y, w=w, h=h),
                confidence=0.8,
            )
        )

    return defects