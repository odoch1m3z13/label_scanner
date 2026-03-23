# workers/scan_worker.py

from __future__ import annotations

import uuid
import time
import json

from loguru import logger
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.schemas import StageResult, ScanVerdict
from app.models.database import ScanResult
from app.services import image_store, reference_store

from app.utils.image import bytes_to_bgr, normalize_image, resize_long_edge

# pipeline stages
from app.pipeline import alignment
from app.pipeline import ocr
from app.pipeline import logo
from app.pipeline import color
from app.pipeline import anomaly
from app.pipeline import diff as decision

settings = get_settings()


# ────────────────────────────────────────────────

async def run_scan(
    image_bytes: bytes,
    label_id: str,
    db: AsyncSession,
) -> dict:
    """
    Full scan pipeline execution.
    Returns API-ready dict.
    """

    scan_id = str(uuid.uuid4())
    t0_total = time.perf_counter()

    logger.info(f"[SCAN {scan_id}] Starting scan for label={label_id}")

    # ── Load reference
    ref_meta = await reference_store.get_label(label_id, db)
    if ref_meta is None:
        raise ValueError(f"Reference label not found: {label_id}")

    ref = bytes_to_bgr(open(ref_meta.image_path, "rb").read())
    ref = normalize_image(ref)

    # ── Load scan image
    scan = normalize_image(resize_long_edge(bytes_to_bgr(image_bytes), 1600))

    # Save raw scan
    image_store.save_scan_image(scan_id, scan)

    all_defects = []
    stages: list[StageResult] = []

    # ─────────────────────────────────────────────
    # STAGE 1 — ALIGNMENT
    # ─────────────────────────────────────────────
    try:
        res = alignment.align(ref, scan)

        scan_aligned = res.warped

        stages.append(StageResult(
            stage="alignment",
            duration_ms=res.duration_ms,
            defects=[],
            metadata={
                "matches": res.num_matches,
                "inlier_ratio": res.inlier_ratio,
                "method": res.method,
            }
        ))

    except Exception as e:
        logger.exception(f"[SCAN {scan_id}] Alignment failed")
        raise RuntimeError("Alignment failed — cannot proceed")

    # ─────────────────────────────────────────────
    # STAGE 2 — OCR
    # ─────────────────────────────────────────────
    try:
        ref_boxes = ocr.dict_to_boxes(ref_meta.ocr_data)
        scan_boxes = ocr.run_ocr(scan_aligned)

        defects, duration = ocr.diff_ocr(ref_boxes, scan_boxes)

        all_defects.extend(defects)

        stages.append(StageResult(
            stage="ocr",
            duration_ms=duration,
            defects=defects,
        ))

    except Exception as e:
        logger.exception(f"[SCAN {scan_id}] OCR failed")
        stages.append(StageResult(stage="ocr", duration_ms=0, defects=[]))

    # ─────────────────────────────────────────────
    # STAGE 3 — LOGO
    # ─────────────────────────────────────────────
    try:
        regions = logo.dict_to_regions(ref_meta.logo_regions)

        defects, duration = logo.compare_logos(ref, scan_aligned, regions)

        all_defects.extend(defects)

        stages.append(StageResult(
            stage="logo",
            duration_ms=duration,
            defects=defects,
        ))

    except Exception as e:
        logger.exception(f"[SCAN {scan_id}] Logo stage failed")
        stages.append(StageResult(stage="logo", duration_ms=0, defects=[]))

    # ─────────────────────────────────────────────
    # STAGE 4 — COLOR
    # ─────────────────────────────────────────────
    try:
        defects, duration = color.compare_colors(
            ref,
            scan_aligned,
            ref_meta.color_profile
        )

        all_defects.extend(defects)

        stages.append(StageResult(
            stage="color",
            duration_ms=duration,
            defects=defects,
        ))

    except Exception as e:
        logger.exception(f"[SCAN {scan_id}] Color stage failed")
        stages.append(StageResult(stage="color", duration_ms=0, defects=[]))

    # ─────────────────────────────────────────────
    # STAGE 5 — ANOMALY
    # ─────────────────────────────────────────────
    try:
        defects, heatmap, duration = anomaly.detect_anomalies(ref, scan_aligned)

        all_defects.extend(defects)

        image_store.save_heatmap(scan_id, heatmap)

        stages.append(StageResult(
            stage="anomaly",
            duration_ms=duration,
            defects=defects,
        ))

    except Exception as e:
        logger.exception(f"[SCAN {scan_id}] Anomaly stage failed")
        stages.append(StageResult(stage="anomaly", duration_ms=0, defects=[]))

    # ─────────────────────────────────────────────
    # STAGE 6 — BARCODE + DECISION
    # ─────────────────────────────────────────────
    try:
        barcode_defects, _ = decision.compare_barcodes(
            ref,
            scan_aligned,
            ref_meta.barcode_values
        )

        all_defects.extend(barcode_defects)

        verdict, ref_ann, scan_ann = decision.decide(
            ref,
            scan_aligned,
            stages,
            all_defects
        )

    except Exception as e:
        logger.exception(f"[SCAN {scan_id}] Decision failed")
        raise RuntimeError("Decision stage failed")

    # ─────────────────────────────────────────────
    # SAVE OUTPUTS
    # ─────────────────────────────────────────────
    ref_path, scan_path = image_store.save_annotated(scan_id, ref_ann, scan_ann)

    duration_total = (time.perf_counter() - t0_total) * 1000

    # ── Stats
    critical = sum(1 for d in all_defects if d.severity == "critical")

    # ── Save to DB
    row = ScanResult(
        id=scan_id,
        label_id=label_id,
        verdict=verdict.value,
        total_defects=len(all_defects),
        critical_defects=critical,
        duration_ms=duration_total,
        stages_json=json.dumps([s.model_dump() for s in stages]),
        defects_json=json.dumps([d.model_dump() for d in all_defects]),
        scan_image_path=str(image_store.image_url(ref_path)),
        annotated_ref_path=str(image_store.image_url(ref_path)),
        annotated_scan_path=str(image_store.image_url(scan_path)),
    )

    db.add(row)
    await db.commit()

    logger.success(f"[SCAN {scan_id}] Completed → {verdict.value}")

    # ── API response
    return {
        "scan_id": scan_id,
        "label_id": label_id,
        "verdict": verdict,
        "total_defects": len(all_defects),
        "critical_defects": critical,
        "stages": [s.model_dump() for s in stages],
        "all_defects": [d.model_dump() for d in all_defects],
        "duration_ms": duration_total,
        "annotated_ref_url": image_store.image_url(ref_path),
        "annotated_scan_url": image_store.image_url(scan_path),
    }