"""
Reference Label Store.

Handles persistence of reference labels:
  – image on disk
  – metadata (OCR, logo regions, colour profile, barcodes) in DB
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import ReferenceLabel
from app.models.schemas import ReferenceMetadata
from app.pipeline import alignment as align_mod
from app.pipeline import color as color_mod
from app.pipeline import logo as logo_mod
from app.pipeline import ocr as ocr_mod
from app.utils.image import (
    bytes_to_bgr,
    normalize_image,
    resize_long_edge,
    save_image,
)
from app.utils.serialization import bboxes_to_dicts

settings = get_settings()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_dir(label_id: str) -> Path:
    d = settings.data_dir / label_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _image_path(label_id: str) -> Path:
    return _label_dir(label_id) / "reference.png"


# ── Public API ────────────────────────────────────────────────────────────────

async def register_label(
    image_bytes: bytes,
    db: AsyncSession,
    label_id: str | None = None,
    template: list | None = None,
) -> ReferenceMetadata:
    """
    Full reference-registration pipeline:
      1. Decode & normalise image
      2. Run OCR
      3. Detect logo regions
      4. Extract colour profile
      5. Decode barcodes
      6. Persist to disk + DB
    """
    if label_id is None:
        label_id = str(uuid.uuid4())

    logger.info(f"Registering reference label: {label_id}")

    # Decode
    img_raw = bytes_to_bgr(image_bytes)
    img = normalize_image(resize_long_edge(img_raw, max_size=1600))
    h, w = img.shape[:2]

    # Stage 4a: OCR
    ocr_boxes = []
    try:
        ocr_boxes = ocr_mod.run_ocr(img)
        ocr_data = ocr_mod.boxes_to_dict(ocr_boxes)
    except Exception as exc:
        logger.warning(f"OCR failed during registration: {exc}")
        ocr_data = []

    template = []

    for b in ocr_boxes:
        if b.confidence <= 0.7:
            continue
            
        if b.box.w * b.box.h < 500:
            continue

        text = b.text.strip().lower()

        if not text:
            continue

        # classify field
        if any(x in text for x in ["exp", "date"]):
            field_type = "expiry"
            strict = True

        elif any(c.isdigit() for c in text):
            field_type = "numeric"
            strict = True

        elif "barcode" in text:
            field_type = "barcode"
            strict = True

        else:
            field_type = "text"
            strict = False

        template.append({
            "name": text.replace(" ", "_"),
            "type": field_type,
            "x": b.box.x,
            "y": b.box.y,
            "w": b.box.w,
            "h": b.box.h,
            "expected_text": b.text,
            "strict": strict,
        })

    # Stage 4b: logo regions
    logo_regions = []
    try:
        logo_regions = logo_mod.detect_logo_regions(img)
        logo_data = logo_mod.regions_to_dict(logo_regions)
    except Exception as exc:
        logger.warning(f"Logo detection failed: {exc}")
        logo_data = []

    # Stage 4c: colour profile
    color_profile = []
    try:
        color_profile = color_mod.extract_color_profile(img)
    except Exception as exc:
        logger.warning(f"Colour profile failed: {exc}")

    # Stage 4d: barcodes
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from PIL import Image as PILImage

        pil = PILImage.fromarray(img[:, :, ::-1])
        barcode_values = [r.data.decode() for r in pyzbar_decode(pil)]
    except Exception:
        barcode_values = []

    # Save image
    img_path = _image_path(label_id)
    save_image(img, img_path)

    # Persist to DB
    row = ReferenceLabel(
        id=label_id,
        created_at=datetime.utcnow(),
        image_path=str(img_path),
        width=w,
        height=h,
        template_json=json.dumps(template),
        ocr_data=json.dumps([
            {
                "text": b.text,
                "confidence": b.confidence,
                "box": {
                    "x": b.box.x,
                    "y": b.box.y,
                    "w": b.box.w,   
                    "h": b.box.h,
                }
            }
            for b in ocr_data
        ]),
        logo_regions=json.dumps(bboxes_to_dicts(logo_regions)),
        color_profile=json.dumps(bboxes_to_dicts(color_profile)),
        barcode_values=json.dumps(barcode_values),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    logger.success(f"Reference label {label_id} registered (OCR boxes: {len(ocr_data)})")

    return ReferenceMetadata(
        label_id=label_id,
        created_at=row.created_at,
        image_path=str(img_path),
        width=w,
        height=h,
        ocr_data=ocr_data,
        logo_regions=bboxes_to_dicts(logo_regions),
        color_profile=bboxes_to_dicts(color_profile),
        barcode_values=barcode_values,
    )


async def get_label(label_id: str, db: AsyncSession) -> ReferenceMetadata | None:
    result = await db.execute(select(ReferenceLabel).where(ReferenceLabel.id == label_id))
    row = result.scalar_one_or_none()
    if row is None:
        return None

    d = row.to_dict()
    return ReferenceMetadata(
        label_id=d["label_id"],
        created_at=datetime.fromisoformat(d["created_at"]),
        image_path=d["image_path"],
        width=d["width"],
        height=d["height"],
        template_json=d.get("template_json"),
        ocr_data=d["ocr_data"],
        logo_regions=d["logo_regions"],
        color_profile=d["color_profile"],
        barcode_values=d["barcode_values"],
    )


async def list_labels(db: AsyncSession) -> list[str]:
    result = await db.execute(select(ReferenceLabel.id))
    return [r[0] for r in result.all()]


async def delete_label(label_id: str, db: AsyncSession) -> bool:
    result = await db.execute(select(ReferenceLabel).where(ReferenceLabel.id == label_id))
    row = result.scalar_one_or_none()
    if row is None:
        return False
    await db.delete(row)
    await db.commit()

    # Remove image from disk
    img_path = _image_path(label_id)
    if img_path.exists():
        img_path.unlink()

    return True