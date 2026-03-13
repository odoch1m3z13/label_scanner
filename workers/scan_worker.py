"""
Celery background worker for async label scanning.

Usage:
    celery -A workers.scan_worker worker --loglevel=info

Enqueue a scan:
    from workers.scan_worker import run_scan_async
    task = run_scan_async.delay(label_id="abc", scan_image_bytes=b"...")
    result = task.get(timeout=30)
"""

from __future__ import annotations

import asyncio
import json

from celery import Celery
from loguru import logger

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "label_scanner",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # one task at a time per worker
    result_expires=3600,            # 1 hour TTL on results
)


# ── Helper to run async code in sync Celery context ───────────────────────────

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="scan_worker.run_scan",
    max_retries=2,
    default_retry_delay=5,
)
def run_scan_async(self, label_id: str, scan_image_b64: str) -> dict:
    """
    Async-safe task that runs the full 6-stage pipeline.

    Args:
        label_id:        Reference label to compare against.
        scan_image_b64:  Base64-encoded scan image bytes.

    Returns:
        Serialised ScanResponse dict.
    """
    import base64

    from app.models.database import AsyncSessionLocal
    from app.pipeline import alignment as align_mod
    from app.pipeline import anomaly as anomaly_mod
    from app.pipeline import color as color_mod
    from app.pipeline import diff as diff_mod
    from app.pipeline import logo as logo_mod
    from app.pipeline import ocr as ocr_mod
    from app.services import image_store, reference_store
    from app.utils.image import bytes_to_bgr, normalize_image, resize_long_edge

    logger.info(f"Worker: scanning against label_id={label_id}")

    image_bytes = base64.b64decode(scan_image_b64)

    async def _pipeline():
        async with AsyncSessionLocal() as db:
            ref_meta = await reference_store.get_label(label_id, db)
            if ref_meta is None:
                raise ValueError(f"Label {label_id} not found")

            from app.utils.image import load_image
            ref_img = load_image(ref_meta.image_path)
            scan_img = normalize_image(resize_long_edge(bytes_to_bgr(image_bytes)))

            # Run all stages (same logic as API endpoint)
            try:
                ar = align_mod.align(ref_img, scan_img)
                scan_aligned = ar.warped
            except Exception:
                scan_aligned = scan_img

            defects = []
            stages = []

            ref_ocr = ocr_mod.dict_to_boxes(ref_meta.ocr_data)
            scan_ocr = ocr_mod.run_ocr(scan_aligned)
            od, _ = ocr_mod.diff_ocr(ref_ocr, scan_ocr)
            defects.extend(od)

            lr = logo_mod.dict_to_regions(ref_meta.logo_regions)
            ld, _ = logo_mod.compare_logos(ref_img, scan_aligned, lr)
            defects.extend(ld)

            cd, _ = color_mod.compare_colors(ref_img, scan_aligned, ref_meta.color_profile)
            defects.extend(cd)

            bd, _ = diff_mod.compare_barcodes(ref_img, scan_aligned, ref_meta.barcode_values)
            defects.extend(bd)

            anom_d, heatmap, _ = anomaly_mod.detect_anomalies(ref_img, scan_aligned)
            defects.extend(anom_d)

            verdict, ann_ref, ann_scan = diff_mod.decide(ref_img, scan_aligned, stages, defects)

            return {
                "verdict": verdict.value,
                "total_defects": len(defects),
                "label_id": label_id,
            }

    try:
        return _run(_pipeline())
    except Exception as exc:
        logger.error(f"Worker task failed: {exc}")
        raise self.retry(exc=exc)