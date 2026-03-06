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
  │  Thread A: Cloud Vision on raw user scan (DOCUMENT_TEXT)       │
  │  Thread B: ORB + RANSAC homography computation                 │
  └────────────────────────────────────────────────────────────────┘
          ↓  coordinate projection (H_inv applied to user boxes)
  ┌─ Phase 2 ──────────────────────────────────────────────────────┐
  │  Thread C: pipeline.diff.run()    — semantic text diff         │
  │  Thread D: pipeline.tamper.run()  — masked visual diff         │
  └────────────────────────────────────────────────────────────────┘
          ↓  combine + schema validation
  CompareResponse → frontend
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
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
    RegisterRequest,
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
    """Startup: initialise storage connections. Shutdown: close them."""
    redis_url    = os.environ.get("REDIS_URL",    "redis://localhost:6379/0")
    postgres_dsn = os.environ.get("POSTGRES_DSN", "postgresql://localhost/labelscanner")

    log.info("Connecting to storage layer…")
    try:
        await cache.init(redis_url, postgres_dsn)
        log.info("Storage layer ready.")
    except NotImplementedError:
        # Storage not yet implemented — run in degraded mode (no caching).
        log.warning("Storage layer not yet implemented — running in stateless mode.")
    except Exception as exc:
        log.error("Storage connection failed: %s", exc)
        # Don't raise — allow server to start so /health returns useful status.

    yield

    log.info("Shutting down storage layer…")
    try:
        await cache.close()
    except Exception:
        pass


app = FastAPI(
    title="Label Diff Scanner",
    version=VERSION,
    lifespan=lifespan,
)


# =============================================================================
#  HELPERS
# =============================================================================

def _load_image(data: bytes, field_name: str = "image") -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{field_name}: could not decode — expected JPEG/PNG/WEBP")
    return img


def _to_b64_preview(img: np.ndarray) -> str:
    h, w = img.shape[:2]
    if max(h, w) > PREVIEW_MAX_PX:
        s   = PREVIEW_MAX_PX / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode()


def _count_diff(diff_entries) -> dict:
    counts = {t: 0 for t in DiffType}
    for d in diff_entries:
        counts[d.type] += 1
    return counts


# =============================================================================
#  ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


@app.post("/register", response_model=RegisterResponse)
async def register(
    label_id:  str         = Form(...),
    ref_image: UploadFile  = File(...),
) -> RegisterResponse:
    """
    Tier 1 — Register a reference label.

    Calls Cloud Vision DOCUMENT_TEXT_DETECTION once, stores the resulting
    SemanticMap in Redis (hot cache) and Postgres (audit record).
    Subsequent calls with the same label_id update the stored map.

    This endpoint is called at label design time, not at scan time.
    """
    image_bytes = await ref_image.read()

    log.info("Registering label '%s' (%d bytes)…", label_id, len(image_bytes))

    # TODO: uncomment once workers/cloud_vision.py is implemented
    # semantic_map = await cloud_vision.extract(image_bytes, label_id)
    # already_existed = await cache.exists(label_id)
    # await cache.set(semantic_map)
    # await registry.register_label(label_id, len(semantic_map.words), len(semantic_map.regions))
    # return RegisterResponse(
    #     label_id=label_id,
    #     word_count=len(semantic_map.words),
    #     region_count=len(semantic_map.regions),
    #     cached=already_existed,
    #     message="Updated" if already_existed else "Registered",
    # )

    raise HTTPException(501, "POST /register not yet implemented — workers/cloud_vision.py pending")


@app.post("/compare", response_model=CompareResponse)
async def compare(
    label_id:   str        = Form(...),
    user_image: UploadFile = File(...),
) -> CompareResponse:
    """
    Tier 2 — Compare a scanned user image against a registered reference label.

    Full parallelised pipeline:
      Phase 1: Cloud Vision (user) + ORB alignment (parallel)
      Phase 2: Text diff + Visual diff (parallel)
    """
    user_bytes = await user_image.read()
    loop       = asyncio.get_event_loop()
    t_start    = loop.time()

    # ── Fetch reference semantic map ─────────────────────────────────────────
    # TODO: uncomment once storage is implemented
    # ref_map = await cache.get(label_id)
    # if ref_map is None:
    #     raise HTTPException(404, f"Label '{label_id}' not registered. POST /register first.")

    # Temporary: raise 501 until storage + CV are implemented
    raise HTTPException(501, "POST /compare not yet implemented — see implementation roadmap")

    # ── Phase 1 (parallel): Cloud Vision on user + ORB homography ────────────
    # ref_bytes  = ref_image_bytes_from_storage   # fetched alongside ref_map
    # user_img   = _load_image(user_bytes, "user_image")
    # ref_img    = _load_image(ref_bytes,  "ref_image")
    #
    # cv_task    = cloud_vision.extract(user_bytes, label_id + ":scan")
    # align_task = loop.run_in_executor(None, align.compute_homography, ref_img, user_img)
    # user_map, alignment = await asyncio.gather(cv_task, align_task)
    #
    # t_phase1 = loop.time() - t_start
    # log.info("Phase 1 complete in %.2fs (CV + ORB)", t_phase1)

    # ── Coordinate projection ─────────────────────────────────────────────────
    # if alignment.H_inv and alignment.status == "ok":
    #     ref_h, ref_w = ref_img.shape[:2]
    #     user_word_boxes = [w.bbox for w in user_map.words]
    #     projected_boxes = align.project_boxes(
    #         user_word_boxes, alignment.H_inv, clip_w=ref_w, clip_h=ref_h
    #     )
    #     for word, proj_box in zip(user_map.words, projected_boxes):
    #         word.bbox = proj_box

    # ── Phase 2 (parallel): text diff + visual diff ───────────────────────────
    # ref_h, ref_w   = ref_img.shape[:2]
    # user_h, user_w = user_img.shape[:2]
    #
    # diff_task   = loop.run_in_executor(None, diff.run,
    #                   ref_map.words, user_map.words,
    #                   ref_h, ref_w, user_h, user_w)
    # tamper_task = loop.run_in_executor(None, tamper.run,
    #                   ref_img, user_img,
    #                   alignment.H, alignment.H_inv,
    #                   ref_map, alignment.inlier_ratio)
    # diff_result, (tamper_ref, tamper_user) = await asyncio.gather(diff_task, tamper_task)
    #
    # t_phase2 = loop.time() - t_start - t_phase1

    # ── Colour changes ────────────────────────────────────────────────────────
    # diff_result, _ = colour.detect_color_changes(
    #     diff_result, ref_img, user_img, alignment.inlier_ratio
    # )

    # ── Barcode comparison ────────────────────────────────────────────────────
    # bc_ref, bc_user = barcode.run(ref_img, user_img, alignment.H, alignment.H_inv)
    # tamper_ref  += bc_ref
    # tamper_user += bc_user

    # ── Layout figure comparison ──────────────────────────────────────────────
    # fig_ref, fig_user = layout.run(
    #     ref_map.regions, user_map.regions,
    #     ref_h, ref_w, user_h, user_w, alignment.H_inv,
    # )
    # tamper_ref  += fig_ref
    # tamper_user += fig_user

    # ── White border delta ────────────────────────────────────────────────────
    # border_boxes = border.run(ref_img, user_img)
    # tamper_user += border_boxes

    # ── Counts ────────────────────────────────────────────────────────────────
    # counts = _count_diff(diff_result)
    # total_time = loop.time() - t_start

    # ── Record scan in audit log ──────────────────────────────────────────────
    # await registry.record_scan(
    #     label_id=label_id,
    #     scan_id=str(uuid.uuid4()),
    #     tamper_detected=bool(tamper_ref or tamper_user),
    #     diff_counts={k.value: v for k, v in counts.items()},
    # )

    # ── Assemble response ─────────────────────────────────────────────────────
    # return CompareResponse(
    #     removed_count=counts[DiffType.removed],
    #     added_count=counts[DiffType.added],
    #     modified_count=counts[DiffType.modified],
    #     color_changed_count=counts[DiffType.color_changed],
    #     unmatched_run_count=counts[DiffType.unmatched_run],
    #     tamper_count_ref=len(tamper_ref),
    #     tamper_count_user=len(tamper_user),
    #     diff=diff_result,
    #     tamper_boxes_ref=tamper_ref,
    #     tamper_boxes_user=tamper_user,
    #     alignment=alignment,
    #     ref_size={"w": ref_w, "h": ref_h},
    #     user_size={"w": user_w, "h": user_h},
    #     ref_word_count=len(ref_map.words),
    #     user_word_count=len(user_map.words),
    #     phase1_time_s=round(t_phase1, 2),
    #     phase2_time_s=round(t_phase2, 2),
    #     total_time_s=round(total_time, 2),
    #     ref_preview=_to_b64_preview(ref_img),
    #     user_preview=_to_b64_preview(user_img),
    # )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    cache_ready = False
    try:
        # TODO: await cache.exists("__health_check__")
        cache_ready = False
    except Exception:
        pass
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