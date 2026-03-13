"""
Stage 5 – Anomaly Detection.

Implements a two-tier approach:
  1. Pixel-level difference map (fast, catches most tampering)
  2. PatchCore-style patch embedding comparison (deep, catches subtle anomalies)
     — only used when torch is available and model is trained.

The output is a per-pixel anomaly score map converted to BoundingBox defects.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings
from app.models.schemas import BoundingBox, ChangeType, Defect, Severity
from app.utils.geometry import mask_to_boxes

settings = get_settings()


# ── Pixel-difference based anomaly map ───────────────────────────────────────

def _pixel_diff_map(ref: np.ndarray, scan: np.ndarray) -> np.ndarray:
    """
    Returns a float32 [0,1] anomaly score map using:
      - Structural similarity (SSIM) local map
      - Gradient magnitude difference
    Combined for robustness.
    """
    from skimage.metrics import structural_similarity as ssim

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    scan_gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # SSIM map (score is 1=identical → invert to get "anomaly")
    _, ssim_map = ssim(ref_gray, scan_gray, full=True, data_range=255)
    anomaly_ssim = (1.0 - ssim_map).astype(np.float32)

    # Gradient difference
    ref_grad = _gradient_magnitude(ref_gray)
    scan_grad = _gradient_magnitude(scan_gray)
    grad_diff = np.abs(ref_grad.astype(np.float32) - scan_grad.astype(np.float32)) / 255.0

    # Colour difference (LAB)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    scan_lab = cv2.cvtColor(scan, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_diff = np.linalg.norm(ref_lab - scan_lab, axis=2) / (255.0 * np.sqrt(3))

    combined = 0.4 * anomaly_ssim + 0.3 * grad_diff + 0.3 * color_diff

    # Smooth to remove sensor noise
    combined = cv2.GaussianBlur(combined, (9, 9), 0)

    return combined


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


# ── PatchCore embedding anomaly (optional deep path) ─────────────────────────

def _patchcore_map(ref: np.ndarray, scan: np.ndarray) -> np.ndarray | None:
    """
    PatchCore-style: extract patch embeddings via ResNet18,
    build a memory bank from the reference, score each scan patch.
    Returns normalised [0,1] anomaly map, or None if torch unavailable.
    """
    try:
        import torch
        import torchvision.transforms as T
        import torchvision.models as models
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Use ResNet18 up to layer2 for patch features
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
    # Extract intermediate feature map
    features_ref: list[np.ndarray] = []
    features_scan: list[np.ndarray] = []

    def _get_patches(img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(rgb).resize((224, 224))
        t = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            # Forward through first layers
            x = backbone.conv1(t)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
        return x.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    fmap_ref = _get_patches(ref)    # shape e.g. (14, 14, 128)
    fmap_scan = _get_patches(scan)

    # Score: cosine distance per patch
    eps = 1e-8
    ref_norm = fmap_ref / (np.linalg.norm(fmap_ref, axis=-1, keepdims=True) + eps)
    scan_norm = fmap_scan / (np.linalg.norm(fmap_scan, axis=-1, keepdims=True) + eps)
    cos_sim = np.sum(ref_norm * scan_norm, axis=-1)  # (H, W)
    anomaly = (1.0 - cos_sim).astype(np.float32)

    # Upsample to original size
    h, w = ref.shape[:2]
    anomaly_up = cv2.resize(anomaly, (w, h), interpolation=cv2.INTER_LINEAR)
    return anomaly_up


# ── Public API ────────────────────────────────────────────────────────────────

def detect_anomalies(
    ref: np.ndarray,
    scan: np.ndarray,
) -> tuple[list[Defect], np.ndarray, float]:
    """
    Run full anomaly detection.

    Returns:
        defects       – list of Defect objects
        heatmap       – uint8 BGR heatmap image for visualisation
        duration_ms
    """
    t0 = time.perf_counter()

    # Pixel diff is always available
    pixel_map = _pixel_diff_map(ref, scan)

    # Try PatchCore; blend if available
    deep_map = _patchcore_map(ref, scan)
    if deep_map is not None:
        score_map = 0.5 * pixel_map + 0.5 * deep_map
    else:
        score_map = pixel_map

    # Normalise to [0, 1]
    vmin, vmax = score_map.min(), score_map.max()
    if vmax > vmin:
        score_norm = (score_map - vmin) / (vmax - vmin)
    else:
        score_norm = score_map

    # Threshold → binary mask
    threshold = settings.anomaly_score_threshold
    binary = (score_norm > threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    boxes = mask_to_boxes(binary, min_area=settings.anomaly_min_area)

    defects: list[Defect] = []
    for box in boxes:
        # Get mean anomaly score in this region
        x, y, w, h = box.to_xywh()
        region_score = float(score_norm[y: y + h, x: x + w].mean())
        severity = Severity.CRITICAL if region_score > 0.75 else Severity.MAJOR
        defects.append(
            Defect(
                change_type=ChangeType.ANOMALY,
                severity=severity,
                description=f"Anomalous region detected (score={region_score:.2f})",
                ref_box=box,
                scan_box=box,
                confidence=region_score,
            )
        )

    # Build colour heatmap for UI
    heatmap_uint8 = (score_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    duration_ms = (time.perf_counter() - t0) * 1000
    return defects, heatmap_color, duration_ms