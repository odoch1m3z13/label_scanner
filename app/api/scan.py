"""
/api/scan – Run the full 6-stage inspection pipeline.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import ScanResult, get_db
from app.models.schemas import (
    ChangeType,
    Defect,
    ScanResponse,
    ScanVerdict,
    Severity,
    StageResult,
)
from app.pipeline import alignment as align_mod
from app.pipeline import anomaly as anomaly_mod
from app.pipeline import color as color_mod
from app.pipeline import diff as diff_mod
from app.pipeline import logo as logo_mod
from app.pipeline import ocr as ocr_mod
from app.services import image_store, reference_store
from app.utils.image import bytes_to_bgr, normalize_image, resize_long_edge

settings = get_settings()
router = APIRouter(prefix="/api", tags=["Scanning"])


# ── Main scan endpoint ────────────────────────────────────────────────────────

@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="Scan a label against a registered reference",
)
async def scan_label(
    file: UploadFile = File(..., description="Scan image to inspect"),
    label_id: str = Form(..., description="Reference label ID to compare against"),
    db: AsyncSession = Depends(get_db),
) -> ScanResponse:
    """
    Full 6-stage pipeline:
      1. Image normalisation
      2. Precision alignment (ORB+RANSAC / LoFTR)
      3. OCR text diff
      4. Logo / graphic comparison (CLIP)
      5. Colour delta (CIEDE2000)
      6. Anomaly detection (pixel + optional PatchCore)
      7. Decision engine
    """
    pipeline_start = time.perf_counter()
    scan_id = str(uuid.uuid4())

    # ── Fetch reference ───────────────────────────────────────────────────────
    ref_meta = await reference_store.get_label(label_id, db)
    if ref_meta is None:
        raise HTTPException(status_code=404, detail=f"Label '{label_id}' not found")

    # ── Decode scan image ─────────────────────────────────────────────────────
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        scan_raw = bytes_to_bgr(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    from app.utils.image import load_image
    ref_img = load_image(ref_meta.image_path)

    # ── Normalise both ────────────────────────────────────────────────────────
    scan_img = normalize_image(resize_long_edge(scan_raw, max_size=1600))
    # Ref was already normalised at registration; just load as-is

    stages: list[StageResult] = []
    all_defects: list[Defect] = []

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Alignment
    # ─────────────────────────────────────────────────────────────────────────
    try:
        align_result = align_mod.align(ref_img, scan_img)
        scan_aligned = align_result.warped
        stages.append(
            StageResult(
                stage="alignment",
                duration_ms=align_result.duration_ms,
                defects=[],
                metadata={
                    "method": align_result.method,
                    "matches": align_result.num_matches,
                },
            )
        )
    except Exception as exc:
        logger.warning(f"Alignment failed, using raw scan: {exc}")
        scan_aligned = scan_img
        stages.append(
            StageResult(
                stage="alignment",
                duration_ms=0,
                defects=[],
                metadata={"error": str(exc)},
            )
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3 / 4a: OCR text diff
    # ─────────────────────────────────────────────────────────────────────────
    try:
        ref_ocr = ocr_mod.dict_to_boxes(ref_meta.ocr_data)
        scan_ocr = ocr_mod.run_ocr(scan_aligned)
        ocr_defects, ocr_ms = ocr_mod.diff_ocr(ref_ocr, scan_ocr)
        all_defects.extend(ocr_defects)
        stages.append(
            StageResult(
                stage="ocr",
                duration_ms=ocr_ms,
                defects=ocr_defects,
                metadata={
                    "ref_boxes": len(ref_ocr),
                    "scan_boxes": len(scan_ocr),
                },
            )
        )
    except Exception as exc:
        logger.warning(f"OCR stage error: {exc}")
        stages.append(StageResult(stage="ocr", duration_ms=0, metadata={"error": str(exc)}))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4b: Logo comparison
    # ─────────────────────────────────────────────────────────────────────────
    try:
        ref_regions = logo_mod.dict_to_regions(ref_meta.logo_regions)
        logo_defects, logo_ms = logo_mod.compare_logos(ref_img, scan_aligned, ref_regions)
        all_defects.extend(logo_defects)
        stages.append(
            StageResult(
                stage="logo",
                duration_ms=logo_ms,
                defects=logo_defects,
                metadata={"regions_checked": len(ref_regions)},
            )
        )
    except Exception as exc:
        logger.warning(f"Logo stage error: {exc}")
        stages.append(StageResult(stage="logo", duration_ms=0, metadata={"error": str(exc)}))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4c: Colour verification
    # ─────────────────────────────────────────────────────────────────────────
    try:
        color_defects, color_ms = color_mod.compare_colors(
            ref_img, scan_aligned, ref_meta.color_profile
        )
        all_defects.extend(color_defects)
        stages.append(
            StageResult(
                stage="color",
                duration_ms=color_ms,
                defects=color_defects,
                metadata={"cells_checked": len(ref_meta.color_profile)},
            )
        )
    except Exception as exc:
        logger.warning(f"Color stage error: {exc}")
        stages.append(StageResult(stage="color", duration_ms=0, metadata={"error": str(exc)}))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4d: Barcode check
    # ─────────────────────────────────────────────────────────────────────────
    try:
        barcode_defects, barcode_ms = diff_mod.compare_barcodes(
            ref_img, scan_aligned, ref_meta.barcode_values
        )
        all_defects.extend(barcode_defects)
        stages.append(
            StageResult(
                stage="barcode",
                duration_ms=barcode_ms,
                defects=barcode_defects,
                metadata={"ref_barcodes": ref_meta.barcode_values},
            )
        )
    except Exception as exc:
        logger.warning(f"Barcode stage error: {exc}")
        stages.append(StageResult(stage="barcode", duration_ms=0, metadata={"error": str(exc)}))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5: Anomaly detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        anomaly_defects, heatmap, anomaly_ms = anomaly_mod.detect_anomalies(ref_img, scan_aligned)
        all_defects.extend(anomaly_defects)
        stages.append(
            StageResult(
                stage="anomaly",
                duration_ms=anomaly_ms,
                defects=anomaly_defects,
                metadata={"regions_flagged": len(anomaly_defects)},
            )
        )
        image_store.save_heatmap(scan_id, heatmap)
    except Exception as exc:
        logger.warning(f"Anomaly stage error: {exc}")
        stages.append(StageResult(stage="anomaly", duration_ms=0, metadata={"error": str(exc)}))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6: Decision engine
    # ─────────────────────────────────────────────────────────────────────────
    verdict, ann_ref, ann_scan = diff_mod.decide(ref_img, scan_aligned, stages, all_defects)

    # Save images
    image_store.save_scan_image(scan_id, scan_aligned)
    ref_ann_path, scan_ann_path = image_store.save_annotated(scan_id, ann_ref, ann_scan)

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    critical_count = sum(1 for d in all_defects if d.severity == Severity.CRITICAL)

    # Persist result to DB
    db_row = ScanResult(
        id=scan_id,
        label_id=label_id,
        verdict=verdict.value,
        total_defects=len(all_defects),
        critical_defects=critical_count,
        duration_ms=total_ms,
        scanned_at=datetime.utcnow(),
        stages_json=json.dumps([s.model_dump() for s in stages]),
        defects_json=json.dumps([d.model_dump() for d in all_defects]),
        annotated_ref_path=str(ref_ann_path),
        annotated_scan_path=str(scan_ann_path),
    )
    db.add(db_row)
    await db.commit()

    logger.info(
        f"Scan {scan_id} | label={label_id} | verdict={verdict.value} "
        f"| defects={len(all_defects)} | {total_ms:.0f}ms"
    )

    return ScanResponse(
        scan_id=scan_id,
        label_id=label_id,
        verdict=verdict,
        total_defects=len(all_defects),
        critical_defects=critical_count,
        stages=stages,
        all_defects=all_defects,
        duration_ms=total_ms,
        scanned_at=datetime.utcnow(),
        annotated_ref_url=image_store.image_url(ref_ann_path),
        annotated_scan_url=image_store.image_url(scan_ann_path),
    )


@router.get("/scans/{scan_id}", summary="Retrieve a previous scan result")
async def get_scan(scan_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import select
    result = await db.execute(select(ScanResult).where(ScanResult.id == scan_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {
        "scan_id": row.id,
        "label_id": row.label_id,
        "verdict": row.verdict,
        "total_defects": row.total_defects,
        "critical_defects": row.critical_defects,
        "duration_ms": row.duration_ms,
        "scanned_at": row.scanned_at.isoformat(),
        "stages": json.loads(row.stages_json),
        "defects": json.loads(row.defects_json),
    }