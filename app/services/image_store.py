"""
Scan Image Store.

Manages storage and URL generation for:
  – raw uploaded scan images
  – annotated reference / scan outputs
  – anomaly heatmaps
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.utils.image import save_image

settings = get_settings()

_SCANS_DIR = settings.data_dir.parent / "scans"
_SCANS_DIR.mkdir(parents=True, exist_ok=True)


def _scan_dir(scan_id: str) -> Path:
    d = _SCANS_DIR / scan_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_scan_image(scan_id: str, img: np.ndarray) -> Path:
    path = _scan_dir(scan_id) / "scan.png"
    save_image(img, path)
    return path


def save_annotated(scan_id: str, ref_ann: np.ndarray, scan_ann: np.ndarray) -> tuple[Path, Path]:
    base = _scan_dir(scan_id)
    ref_path = base / "annotated_ref.png"
    scan_path = base / "annotated_scan.png"
    save_image(ref_ann, ref_path)
    save_image(scan_ann, scan_path)
    return ref_path, scan_path


def save_heatmap(scan_id: str, heatmap: np.ndarray) -> Path:
    path = _scan_dir(scan_id) / "heatmap.png"
    save_image(heatmap, path)
    return path


def image_url(path: Path | str) -> str:
    """Convert absolute path to a URL served by the /data static mount."""
    path = Path(path)
    # Find 'data' in the path and return everything from there
    parts = path.parts
    try:
        data_idx = [p.lower() for p in parts].index('data')
        rel = Path(*parts[data_idx:])
        return f"/{rel.as_posix()}"
    except ValueError:
        return f"/data/{path.name}"