"""
Stage 4b – Logo & Graphic Region Verification.

Uses CLIP (ViT-B/32) to generate image embeddings for detected
logo/graphic regions, then compares via cosine similarity.

Falls back to ORB-based structural similarity if torch is unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.config import get_settings
from app.models.schemas import BoundingBox, ChangeType, Defect, Severity

settings = get_settings()


# ── CLIP embedding helper ─────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_device = None


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        import torch
        import open_clip

        _clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=settings.clip_pretrained,
            device=_clip_device,
        )
        _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_device


def _clip_embed(crop: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image as PILImage

    model, preprocess, device = _get_clip()
    pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── ORB-based fallback ────────────────────────────────────────────────────────

def _orb_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=500)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    _, des_a = orb.detectAndCompute(gray_a, None)
    _, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < 60]
    return min(1.0, len(good) / max(len(des_a), len(des_b), 1) * 5)


# ── Region detection (simple contour-based) ──────────────────────────────────

def detect_logo_regions(img: np.ndarray) -> list[BoundingBox]:
    """
    Detect candidate logo / graphic regions using edge density.
    Real deployment would use a segmentation model here.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    min_area = (h * w) * 0.005   # at least 0.5% of image
    max_area = (h * w) * 0.4     # at most 40%

    boxes: list[BoundingBox] = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append(BoundingBox(x=x, y=y, w=bw, h=bh))

    return boxes


def regions_to_dict(boxes: list[BoundingBox]) -> list[dict[str, Any]]:
    return [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in boxes]


def dict_to_regions(data: list[dict]) -> list[BoundingBox]:
    return [BoundingBox(x=d["x"], y=d["y"], w=d["w"], h=d["h"]) for d in data]


# ── Crop helper ───────────────────────────────────────────────────────────────

def _crop(img: np.ndarray, box: BoundingBox) -> np.ndarray:
    x, y, w, h = box.to_xywh()
    crop = img[y: y + h, x: x + w]
    if crop.size == 0:
        raise ValueError(f"Empty crop for box {box}")
    return crop


# ── Public API ────────────────────────────────────────────────────────────────

def compare_logos(
    ref: np.ndarray,
    scan: np.ndarray,
    ref_regions: list[BoundingBox],
) -> tuple[list[Defect], float]:
    """
    Compare each stored reference logo region against the aligned scan.
    Returns (defects, duration_ms).
    """
    t0 = time.perf_counter()
    defects: list[Defect] = []

    use_clip = True
    try:
        import torch  # noqa: F401
        import open_clip  # noqa: F401
    except ImportError:
        use_clip = False

    for region in ref_regions:
        try:
            ref_crop = _crop(ref, region)
            scan_crop = _crop(scan, region)
            # Resize scan crop to match ref crop for fair comparison
            scan_crop = cv2.resize(scan_crop, (ref_crop.shape[1], ref_crop.shape[0]))
        except Exception:
            continue

        if use_clip:
            try:
                ref_emb = _clip_embed(ref_crop)
                scan_emb = _clip_embed(scan_crop)
                similarity = _cosine(ref_emb, scan_emb)
            except Exception:
                similarity = _orb_similarity(ref_crop, scan_crop)
        else:
            similarity = _orb_similarity(ref_crop, scan_crop)

        if similarity < settings.logo_similarity_threshold:
            defects.append(
                Defect(
                    change_type=ChangeType.LOGO,
                    severity=Severity.MAJOR,
                    description=f"Logo/graphic mismatch (similarity={similarity:.3f})",
                    ref_box=region,
                    scan_box=region,
                    confidence=1.0 - similarity,
                )
            )

    duration_ms = (time.perf_counter() - t0) * 1000
    return defects, duration_ms