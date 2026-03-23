from __future__ import annotations
import time
import cv2
import numpy as np

from app.models.schemas import Defect, ChangeType, Severity
from app.utils.geometry import mask_to_boxes


def _diff(ref, scan):
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    return np.abs(g1.astype(np.float32) - g2.astype(np.float32)) / 255.0


def detect_anomalies(ref, scan):
    t0 = time.perf_counter()

    diff = _diff(ref, scan)

    mean, std = diff.mean(), diff.std()
    thresh = mean + 1.5*std

    binary = (diff > thresh).astype(np.uint8)*255

    boxes = mask_to_boxes(binary)

    defects = []
    for b in boxes:
        x,y,w,h = b.to_xywh()
        score = float(diff[y:y+h, x:x+w].mean())

        defects.append(Defect(
            change_type=ChangeType.ANOMALY,
            severity=Severity.MAJOR,
            description=f"Anomaly score={score:.2f}",
            ref_box=b,
            scan_box=b,
            confidence=score
        ))

    heatmap = (diff*255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return defects, heatmap, (time.perf_counter()-t0)*1000