"""
Geometric utilities: IoU, merging, coordinate transforms, drawing.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import BoundingBox


# ── IoU ───────────────────────────────────────────────────────────────────────

def iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.to_xyxy()
    bx1, by1, bx2, by2 = b.to_xyxy()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = a.area + b.area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ── Contour → BoundingBox ─────────────────────────────────────────────────────

def contour_to_box(contour: np.ndarray) -> BoundingBox:
    x, y, w, h = cv2.boundingRect(contour)
    return BoundingBox(x=x, y=y, w=w, h=h)


def mask_to_boxes(
    mask: np.ndarray,
    min_area: int = 100,
    dilate_px: int = 4,
) -> list[BoundingBox]:
    """Convert a binary mask to a list of bounding boxes (after morphological cleanup)."""
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            boxes.append(contour_to_box(c))
    return boxes


# ── NMS ───────────────────────────────────────────────────────────────────────

def nms_boxes(boxes: list[BoundingBox], iou_threshold: float = 0.4) -> list[BoundingBox]:
    """Simple greedy NMS (largest area first)."""
    boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
    kept: list[BoundingBox] = []
    for candidate in boxes:
        if all(iou(candidate, k) < iou_threshold for k in kept):
            kept.append(candidate)
    return kept


# ── Drawing ───────────────────────────────────────────────────────────────────

GREEN = (0, 200, 0)
RED = (0, 0, 220)
ORANGE = (0, 140, 255)
YELLOW = (0, 220, 220)

_SEVERITY_COLORS = {
    "critical": RED,
    "major": ORANGE,
    "minor": YELLOW,
}


def draw_boxes(
    img: np.ndarray,
    boxes: list[BoundingBox],
    color: tuple[int, int, int] = RED,
    thickness: int = 2,
    labels: list[str] | None = None,
) -> np.ndarray:
    out = img.copy()
    for i, box in enumerate(boxes):
        x, y, w, h = box.to_xywh()
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if labels and i < len(labels):
            _put_label(out, labels[i], x, y, color)
    return out


def _put_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    bg_y = max(0, y - th - 4)
    cv2.rectangle(img, (x, bg_y), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def annotate_image(
    img: np.ndarray,
    defects: list,     # list[Defect] — typed loosely to avoid circular import
    use_scan_box: bool = True,
) -> np.ndarray:
    """Draw all defect bounding boxes on an image with severity colours."""
    out = img.copy()
    for d in defects:
        box = d.scan_box if use_scan_box else d.ref_box
        if box is None:
            continue
        color = _SEVERITY_COLORS.get(d.severity, RED)
        label = f"{d.change_type}:{d.severity}"
        x, y, w, h = box.to_xywh()
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        _put_label(out, label, x, y, color)
    return out


# ── Coordinate scaling ────────────────────────────────────────────────────────

def scale_box(box: BoundingBox, sx: float, sy: float) -> BoundingBox:
    return BoundingBox(
        x=int(box.x * sx),
        y=int(box.y * sy),
        w=max(1, int(box.w * sx)),
        h=max(1, int(box.h * sy)),
    )