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
    import json
    template = json.loads(ref_meta.template_json or "[]")

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
    # STAGE 3: Template text diff
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from app.pipeline.template import check_template_text
        t0 = time.perf_counter()
        
        defects = check_template_text(ref_img, scan_aligned, template)
        
        duration = (time.perf_counter() - t0) * 1000
        
        all_defects.extend(defects)
        
        stages.append(StageResult(
            stage="template_text",
            duration_ms=duration,
            defects=defects,
        ))
    except Exception as exc:
        logger.warning(f"Template stage error: {exc}")
        stages.append(StageResult(stage="template_text", duration_ms=0, metadata={"error": str(exc)}))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3b: Negative space detection
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from app.pipeline.template import detect_unexpected_changes
        t0 = time.perf_counter()
        
        extra_defects = detect_unexpected_changes(ref_img, scan_aligned, template)
        
        duration = (time.perf_counter() - t0) * 1000
        
        all_defects.extend(extra_defects)
        
        stages.append(StageResult(
            stage="unexpected_detection",
            duration_ms=duration,
            defects=extra_defects,
        ))
    except Exception as exc:
        logger.warning(f"Unexpected detection stage error: {exc}")
        stages.append(StageResult(stage="unexpected_detection", duration_ms=0, metadata={"error": str(exc)}))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4b: Logo comparison
    # ─────────────────────────────────────────────────────────────────────────
    # Disabled for template mode
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4c: Colour verification
    # ─────────────────────────────────────────────────────────────────────────
    # Disabled for template mode
    # ─────────────────────────────────────────────────────────────────────────
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
    # anomaly stage disabled for template mode
    # ─────────────────────────────────────────────────────────────────────────
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