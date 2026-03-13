"""
Image utility helpers: load, save, normalise, resize, encode.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ── Load / save ───────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """Load image as BGR numpy array (raises on failure)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Convert raw image bytes → BGR ndarray."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def bgr_to_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def bgr_to_base64(img: np.ndarray, ext: str = ".jpg") -> str:
    raw = bgr_to_bytes(img, ext)
    return base64.b64encode(raw).decode()


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Stage-1 normalisation pipeline:
      1. CLAHE on L-channel (LAB) for consistent brightness
      2. Bilateral filter to denoise while preserving edges
      3. Gamma correction (auto)
    """
    # CLAHE on luminance
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Bilateral denoise
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Auto gamma
    img = _auto_gamma(img)

    return img


def _auto_gamma(img: np.ndarray) -> np.ndarray:
    mean_lum = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255.0
    if mean_lum <= 0:
        return img
    gamma = np.log(0.5) / np.log(mean_lum)
    gamma = float(np.clip(gamma, 0.4, 2.5))
    lut = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, lut)


# ── Resize / scale ────────────────────────────────────────────────────────────

def resize_to_match(src: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize src to match target's (h, w)."""
    th, tw = target.shape[:2]
    if src.shape[:2] == (th, tw):
        return src
    return cv2.resize(src, (tw, th), interpolation=cv2.INTER_LANCZOS4)


def resize_long_edge(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ── Conversion helpers ────────────────────────────────────────────────────────

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(img))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)