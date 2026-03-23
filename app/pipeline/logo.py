from __future__ import annotations
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.models.schemas import Defect, ChangeType, Severity, BoundingBox
from app.utils.common import crop

def compare_logos(ref, scan, regions):
    t0 = time.perf_counter()
    defects = []

    for region in regions:
        ref_crop = crop(ref, region)
        scan_crop = crop(scan, region)

        if ref_crop is None or scan_crop is None:
            continue

        scan_crop = cv2.resize(scan_crop, (ref_crop.shape[1], ref_crop.shape[0]))

        gray1 = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(scan_crop, cv2.COLOR_BGR2GRAY)

        score = ssim(gray1, gray2)

        if score < 0.75:
            defects.append(Defect(
                change_type=ChangeType.LOGO,
                severity=Severity.MAJOR,
                description=f"Logo mismatch (ssim={score:.2f})",
                ref_box=region,
                scan_box=region,
                confidence=1-score
            ))

    return defects, (time.perf_counter()-t0)*1000

def dict_to_regions(data):
    """
    Convert stored regions into BoundingBox list
    """
    if not data:
        return []

    regions = []
    for r in data:
        try:
            regions.append(BoundingBox(
                x=r["x"],
                y=r["y"],
                w=r["w"],
                h=r["h"],
            ))
        except Exception:
            continue

    return regions

def regions_to_dict(regions):
    """
    Convert BoundingBox list → JSON-safe dict list
    """
    out = []
    for r in regions:
        out.append({
            "x": r.x,
            "y": r.y,
            "w": r.w,
            "h": r.h,
        })
    return out    

def detect_logo_regions(img):
    """
    TEMP: fallback → whole image as one region
    """
    h, w = img.shape[:2]
    return [BoundingBox(x=0, y=0, w=w, h=h)]