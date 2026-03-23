from __future__ import annotations
import time
import cv2
import numpy as np

from app.models.schemas import Defect, ChangeType, Severity, BoundingBox
from app.utils.common import crop
from app.config import get_settings

settings = get_settings()


def compare_colors(ref, scan, regions):
    t0 = time.perf_counter()
    defects = []

    for region in regions:
        ref_crop = crop(ref, region)
        scan_crop = crop(scan, region)

        if ref_crop is None or scan_crop is None:
            continue

        ref_lab = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        scan_lab = cv2.cvtColor(scan_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

        delta = np.linalg.norm(ref_lab - scan_lab, axis=2)

        ratio = (delta > settings.color_delta_e_threshold).mean()

        if ratio > 0.1:
            defects.append(Defect(
                change_type=ChangeType.COLOR,
                severity=Severity.MAJOR,
                description=f"{ratio*100:.1f}% color shift",
                ref_box=region,
                scan_box=region,
                confidence=ratio
            ))

    return defects, (time.perf_counter()-t0)*1000

def extract_color_profile(img):
    """
    TEMP: treat whole image as one region
    """
    h, w = img.shape[:2]
    return [BoundingBox(x=0, y=0, w=w, h=h)]