"""
main.py — Label Diff Scanner v5.0
FastAPI application. Thin orchestration only — all logic lives in pipeline/ and workers/.

Endpoints:
  POST /register          Tier 1: Register a reference label. Calls Cloud Vision once,
                          stores SemanticMap in Redis + Postgres.
  POST /compare           Tier 2: Scan a user image against a registered reference.
  GET  /health            Liveness + readiness check.
  GET  /                  Serve frontend SPA.

Scan-time parallelisation (asyncio.gather):

  ┌─ Phase 1 ──────────────────────────────────────────────────────┐
  │  Coroutine A: Cloud Vision on raw user scan (DOCUMENT_TEXT)    │
  │  Thread    B: ORB + RANSAC homography computation              │
  └────────────────────────────────────────────────────────────────┘
          ↓  coordinate projection (H_inv applied to user word boxes)
  ┌─ Phase 2 ──────────────────────────────────────────────────────┐
  │  Thread C: pipeline.diff.run()    — semantic text diff         │
  │  Thread D: pipeline.tamper.run()  — masked visual diff         │
  └────────────────────────────────────────────────────────────────┘
          ↓  colour / barcode / layout / border (serial, fast)
  CompareResponse → frontend

Why serial for colour/barcode/layout/border:
  These four steps share the already-computed diff and tamper lists as inputs or
  outputs. Parallelising them would require immutable snapshots, adding complexity
  with negligible wall-clock gain (each step is <50 ms on CPU).

Reference image storage:
  POST /register stores the raw reference image bytes as BYTEA in Postgres via
  registry.store_ref_image(). POST /compare retrieves them via registry.get_ref_image()
  so the caller never needs to re-upload the reference at scan time.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

import pipeline.align   as align
import pipeline.barcode as barcode
import pipeline.border  as border
import pipeline.colour  as colour
import pipeline.diff    as diff
import pipeline.layout  as layout
import pipeline.tamper  as tamper
import storage.semantic_cache as cache
import storage.registry       as registry
import workers.cloud_vision   as cloud_vision
from config import PREVIEW_JPEG_QUALITY, PREVIEW_MAX_PX
from models.schemas import (
    AlignmentResult,
    CompareResponse,
    DiffType,
    HealthResponse,
    RegisterResponse,
    SemanticMap,
    TamperBox,
    TamperSource,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

VERSION = "5.0"


# =============================================================================
#  LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: connect Redis + Postgres (which also wires registry via
    semantic_cache.init → registry.init).

    Shutdown: close all connections cleanly.

    The server starts even when storage is unavailable — /health will report
    cache_ready=False, and /register + /compare will return 503 until storage
    comes back. This lets the process come up and be inspected before
    dependencies are healthy.
    """
    redis_url    = os.environ.get("REDIS_URL",    "redis://localhost:6379/0")
    postgres_dsn = os.environ.get("POSTGRES_DSN", "postgresql://localhost/labelscanner")

    log.info("Connecting to storage layer…")
    try:
        # semantic_cache.init() also calls registry.init() internally to share
        # the same asyncpg pool (see storage/semantic_cache.py).
        await cache.init(redis_url, postgres_dsn)
        log.info("Storage layer ready.")
    except Exception as exc:
        log.error("Storage connection failed: %s — running in degraded mode.", exc)
        # Do NOT raise — let the process start so /health returns useful status.

    yield

    log.info("Shutting down storage layer…")
    try:
        # semantic_cache.close() also calls registry.close() internally.
        await cache.close()
    except Exception as exc:
        log.warning("Error during storage shutdown: %s", exc)


app = FastAPI(
    title="Label Diff Scanner",
    version=VERSION,
    lifespan=lifespan,
)


# =============================================================================
#  HELPERS
# =============================================================================

def _load_image(data: bytes, field_name: str = "image") -> np.ndarray:
    """Decode raw bytes into a BGR numpy array. Raises HTTP 400 on failure."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            400, f"{field_name}: could not decode — expected JPEG / PNG / WEBP"
        )
    return img


def _to_b64_preview(img: np.ndarray) -> str:
    """
    Thumbnail `img` to PREVIEW_MAX_PX on the longest edge, encode as JPEG,
    return base64 string for the frontend preview panel.
    """
    h, w = img.shape[:2]
    if max(h, w) > PREVIEW_MAX_PX:
        s   = PREVIEW_MAX_PX / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode()


def _count_diff(diff_entries: list) -> dict:
    """Return per-DiffType counts as a dict keyed by DiffType enum member."""
    counts: dict[DiffType, int] = {t: 0 for t in DiffType}
    for entry in diff_entries:
        counts[entry.type] += 1
    return counts


def _require_storage(label_id: str | None = None) -> None:
    """
    Raise HTTP 503 if the storage layer is not initialised.
    Used at the top of /register and /compare to give a clear error rather
    than a cryptic RuntimeError from inside the cache module.
    """
    import storage.semantic_cache as _cache
    if _cache._redis is None or _cache._pg_pool is None:
        detail = (
            "Storage layer unavailable — Redis / Postgres not connected. "
            "Check REDIS_URL and POSTGRES_DSN environment variables."
        )
        if label_id:
            detail += f" (label_id={label_id})"
        raise HTTPException(503, detail)


# =============================================================================
#  ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the frontend single-page application."""
    index = Path("static/index.html")
    if not index.exists():
        raise HTTPException(404, "Frontend not found — static/index.html missing.")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/register", response_model=RegisterResponse)
async def register(
    label_id:  str        = Form(..., description="Unique SKU / label identifier"),
    ref_image: UploadFile = File(..., description="Reference label image (JPEG/PNG/WEBP)"),
) -> RegisterResponse:
    """
    Tier 1 — Register a reference label.

    Calls Cloud Vision DOCUMENT_TEXT_DETECTION once per registration.
    The resulting SemanticMap is stored in Redis (hot cache) and Postgres
    (system of record). The raw image bytes are also persisted so that
    POST /compare can retrieve the reference pixels without a re-upload.

    Re-registering an existing label_id updates both the SemanticMap and
    the stored image bytes. The `cached` field in the response indicates
    whether this was a new registration (False) or an update (True).
    """
    _require_storage(label_id)

    image_bytes = await ref_image.read()
    if not image_bytes:
        raise HTTPException(400, "ref_image: empty file — no bytes received.")

    log.info("POST /register  label_id=%s  size=%d bytes", label_id, len(image_bytes))

    # ── Cloud Vision extraction ────────────────────────────────────────────────
    try:
        semantic_map: SemanticMap = await cloud_vision.extract(image_bytes, label_id)
    except ValueError as exc:
        raise HTTPException(400, f"Cloud Vision rejected the image: {exc}") from exc
    except Exception as exc:
        log.exception("Cloud Vision extraction failed for label '%s'.", label_id)
        raise HTTPException(502, f"Cloud Vision error: {exc}") from exc

    # ── Persistence ────────────────────────────────────────────────────────────
    # Check existence BEFORE writing so the response can report new vs update.
    already_existed = await cache.exists(label_id)

    # Write SemanticMap to Redis (hot) + Postgres (record).
    await cache.set(semantic_map)

    # Update labels table metadata columns (word_count, region_count).
    await registry.register_label(
        label_id,
        word_count=len(semantic_map.words),
        region_count=len(semantic_map.regions),
    )

    # Store raw reference image bytes for pixel-level /compare comparisons.
    await registry.store_ref_image(label_id, image_bytes)

    log.info(
        "Registered label '%s'  words=%d  regions=%d  update=%s",
        label_id, len(semantic_map.words), len(semantic_map.regions), already_existed,
    )

    return RegisterResponse(
        label_id=label_id,
        word_count=len(semantic_map.words),
        region_count=len(semantic_map.regions),
        cached=already_existed,
        message="Updated" if already_existed else "Registered",
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(
    label_id:   str        = Form(..., description="Registered label identifier"),
    user_image: UploadFile = File(..., description="Scanned user image (JPEG/PNG/WEBP)"),
) -> CompareResponse:
    """
    Tier 2 — Compare a scanned user image against a registered reference label.

    Full parallelised pipeline:

      Phase 1 (parallel):
        - Cloud Vision DOCUMENT_TEXT_DETECTION on the raw user scan.
        - ORB + RANSAC homography between reference and user images.

      Phase 2 (parallel):
        - Semantic text diff (fuzzy LCS + spatial second pass).
        - Masked pixel-level visual diff (SSIM + LAB delta).

      Serial post-processing:
        - Colour ink change detection (LAB chroma calibration).
        - Barcode HOG comparison.
        - Layout figure IoU matching.
        - White border / padding delta.
        - Scan audit record persisted to Postgres.
    """
    _require_storage(label_id)

    user_bytes = await user_image.read()
    if not user_bytes:
        raise HTTPException(400, "user_image: empty file — no bytes received.")

    loop    = asyncio.get_event_loop()
    t_start = loop.time()

    # ── Fetch reference data ───────────────────────────────────────────────────
    ref_map   = await cache.get(label_id)
    ref_bytes = await registry.get_ref_image(label_id)

    if ref_map is None:
        raise HTTPException(
            404,
            f"Label '{label_id}' is not registered. POST /register first.",
        )
    if ref_bytes is None:
        raise HTTPException(
            404,
            f"Reference image bytes for '{label_id}' are missing — "
            "please re-register this label.",
        )

    log.info(
        "POST /compare  label_id=%s  user_size=%d bytes", label_id, len(user_bytes)
    )

    # ── Decode images ──────────────────────────────────────────────────────────
    user_img = _load_image(user_bytes, "user_image")
    ref_img  = _load_image(ref_bytes,  "ref_image")
    ref_h,  ref_w  = ref_img.shape[:2]
    user_h, user_w = user_img.shape[:2]

    # ── Phase 1 (parallel): Cloud Vision on user + ORB alignment ──────────────
    # Cloud Vision is async (gRPC via thread executor inside cloud_vision.extract).
    # compute_homography is CPU-bound; we push it to the default thread executor.
    cv_task    = cloud_vision.extract(user_bytes, label_id + ":scan")
    align_task = loop.run_in_executor(
        None, align.compute_homography, ref_img, user_img
    )
    user_map, alignment = await asyncio.gather(cv_task, align_task)

    t_after_phase1 = loop.time()
    t_phase1       = t_after_phase1 - t_start
    log.info("Phase 1 complete in %.2fs (CV + ORB)", t_phase1)

    # ── Coordinate projection ──────────────────────────────────────────────────
    # Project user word bboxes and polygons from user coordinate space into
    # ref coordinate space via H_inv. This keeps Cloud Vision running on the
    # clean unwarped user scan while giving the text diff a shared coordinate
    # system. Only done when alignment succeeded.
    if alignment.status == "ok" and alignment.H_inv is not None:
        user_words_proj = align.project_words(
            user_map.words, alignment.H_inv, clip_w=ref_w, clip_h=ref_h
        )
    else:
        log.warning(
            "Alignment status '%s' — text diff running in unaligned coordinate space.",
            alignment.status,
        )
        user_words_proj = user_map.words

    # ── Phase 2 (parallel): text diff + visual tamper ─────────────────────────
    # Both are CPU-bound; each goes to the default thread executor.
    diff_task = loop.run_in_executor(
        None,
        diff.run,
        ref_map.words,
        user_words_proj,
        ref_h, ref_w,
        user_h, user_w,
    )
    tamper_task = loop.run_in_executor(
        None,
        tamper.run,
        ref_img,
        user_img,
        alignment.H,
        alignment.H_inv,
        ref_map,
        alignment.inlier_ratio,
    )
    diff_result, tamper_pair = await asyncio.gather(diff_task, tamper_task)
    tamper_ref, tamper_user  = tamper_pair   # tuple[list[TamperBox], list[TamperBox]]

    t_after_phase2 = loop.time()
    t_phase2       = t_after_phase2 - t_after_phase1
    log.info("Phase 2 complete in %.2fs (diff + tamper)", t_phase2)

    # ── Colour ink change detection ────────────────────────────────────────────
    # Runs on the diff list + both images. Promotes match/modified → color_changed
    # where the calibrated LAB chroma delta exceeds the threshold.
    diff_result, _ = colour.detect_color_changes(
        diff_result, ref_img, user_img, alignment.inlier_ratio
    )

    # ── Barcode HOG comparison ─────────────────────────────────────────────────
    bc_ref, bc_user = barcode.run(
        ref_img, user_img, alignment.H, alignment.H_inv
    )
    tamper_ref  = tamper_ref  + bc_ref
    tamper_user = tamper_user + bc_user

    # ── Layout figure IoU matching ─────────────────────────────────────────────
    fig_ref, fig_user = layout.run(
        ref_map.regions, user_map.regions,
        ref_h, ref_w,
        user_h, user_w,
        alignment.H_inv,
    )
    tamper_ref  = tamper_ref  + fig_ref
    tamper_user = tamper_user + fig_user

    # ── White border / padding delta ───────────────────────────────────────────
    border_boxes = border.run(ref_img, user_img)
    tamper_user  = tamper_user + border_boxes

    # ── Counts ─────────────────────────────────────────────────────────────────
    counts = _count_diff(diff_result)

    white_border_count = sum(
        1 for tb in tamper_user if tb.source == TamperSource.white_border
    )
    pp_figure_count = sum(
        1 for tb in (tamper_ref + tamper_user) if tb.source == TamperSource.layout
    )

    total_time = loop.time() - t_start
    log.info(
        "Compare complete — label=%s  total=%.2fs  "
        "diffs(rm=%d add=%d mod=%d col=%d)  tamper(ref=%d user=%d)",
        label_id, total_time,
        counts[DiffType.removed], counts[DiffType.added],
        counts[DiffType.modified], counts[DiffType.color_changed],
        len(tamper_ref), len(tamper_user),
    )

    # ── Persist scan audit record ──────────────────────────────────────────────
    tamper_detected = bool(tamper_ref or tamper_user)
    try:
        await registry.record_scan(
            label_id=label_id,
            scan_id=str(uuid.uuid4()),
            tamper_detected=tamper_detected,
            diff_counts={k.value: v for k, v in counts.items()},
        )
    except Exception:
        # Non-fatal — audit failure must never cause a scan to fail.
        log.exception("record_scan failed for label_id=%s — scan result still returned.", label_id)

    # ── Assemble response ──────────────────────────────────────────────────────
    return CompareResponse(
        removed_count=counts[DiffType.removed],
        added_count=counts[DiffType.added],
        modified_count=counts[DiffType.modified],
        color_changed_count=counts[DiffType.color_changed],
        unmatched_run_count=counts[DiffType.unmatched_run],
        tamper_count_ref=len(tamper_ref),
        tamper_count_user=len(tamper_user),
        white_border_count=white_border_count,
        pp_figure_count=pp_figure_count,
        diff=diff_result,
        tamper_boxes_ref=tamper_ref,
        tamper_boxes_user=tamper_user,
        alignment=alignment,
        ref_size={"w": ref_w, "h": ref_h},
        user_size={"w": user_w, "h": user_h},
        ref_word_count=len(ref_map.words),
        user_word_count=len(user_map.words),
        phase1_time_s=round(t_phase1, 2),
        phase2_time_s=round(t_phase2, 2),
        total_time_s=round(total_time, 2),
        ref_preview=_to_b64_preview(ref_img),
        user_preview=_to_b64_preview(user_img),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness + readiness check.

    cache_ready reflects whether Redis + Postgres are reachable.
    A False value means /register and /compare will return 503.
    """
    cache_ready = False
    try:
        cache_ready = await cache.exists("__health_check__")
    except Exception:
        pass   # storage not initialised or unreachable — cache_ready stays False

    return HealthResponse(
        status="ok",
        workers_ready=True,   # no process pool needed — Cloud Vision is async
        cache_ready=cache_ready,
        version=VERSION,
    )


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)