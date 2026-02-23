"""
Label Diff Scanner — FastAPI + PaddleOCR + OpenCV backend
Run:  python main.py
Open: http://localhost:8000
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Skip PaddleOCR's slow connectivity check on every startup
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# =============================================================================
#  WORKER-PROCESS CODE
#  Must be top-level (module scope) so ProcessPoolExecutor can pickle them.
# =============================================================================

_paddle: object = None   # lives only in worker processes

# Half the logical cores per worker so both workers don't fight for threads
_THREADS_PER_WORKER = max(1, (os.cpu_count() or 2) // 2)


def _worker_init() -> None:
    """
    Runs once when each worker process starts.
    Loads PaddleOCR into module-global _paddle so it is reused across
    every request — zero reload cost per scan.

    Pinned to PaddleOCR 2.7.3 + PaddlePaddle 2.6.2 — the stable 2.x line
    with the full constructor API and no PIR executor issues.
    """
    global _paddle

    # Tell PaddlePaddle how many threads this worker may use
    os.environ["OMP_NUM_THREADS"]    = str(_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"]    = str(_THREADS_PER_WORKER)
    os.environ["PADDLE_NUM_THREADS"] = str(_THREADS_PER_WORKER)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    # Increase download timeout and disable parallel downloads
    os.environ["PADDLE_DOWNLOAD_TIMEOUT"] = "600"
    os.environ["PYTHONUNBUFFERED"] = "1"

    from paddleocr import PaddleOCR  # noqa: PLC0415

    # PaddleOCR 2.7.x — full kwargs available, stable on CPU
    _paddle = PaddleOCR(
        use_angle_cls=True,   # handle rotated text lines
        lang="en",
        use_gpu=False,
        cpu_threads=_THREADS_PER_WORKER,
        show_log=False,
        enable_mkldnn=True,   # Intel MKL-DNN — ~30% faster on x86 CPUs
        warmup=False,         # Skip warmup to speed up initialization
    )

    logging.getLogger(__name__).info(
        "Worker PID %d: PaddleOCR ready (%d threads)", os.getpid(), _THREADS_PER_WORKER
    )


def _parse_paddle_result(raw: object) -> list[tuple[list, str, float]]:
    """
    Parse PaddleOCR result defensively across 2.x and 3.x return formats.

    2.x format:  raw = [ [[[x,y],[x,y],[x,y],[x,y]], (text, conf)], ... ]
                 (outer list per page, inner list per line)

    3.x format:  raw = [ ResultObject ]  where each ResultObject is iterable
                 and yields (box, (text, conf)) — identical shape but wrapped
                 in a result class rather than a plain list.
                 Some 3.x builds also expose .boxes / .txts / .scores attrs.
    """
    results: list[tuple[list, str, float]] = []

    if raw is None:
        return results

    # Unwrap outer page list — always index 0 for single-image input
    try:
        lines = raw[0]  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        lines = raw

    if lines is None:
        return results

    for line in lines:
        if line is None:
            continue

        # ── Try old 2.x / compatible 3.x format: (pts, (text, conf)) ──────
        try:
            pts_raw, text_conf = line
            text, conf = text_conf
            pts = [[float(p[0]), float(p[1])] for p in pts_raw]
            results.append((pts, str(text).strip(), float(conf)))
            continue
        except (TypeError, ValueError, IndexError):
            pass

        # ── Try 3.x attribute-based format ──────────────────────────────────
        try:
            pts  = [[float(p[0]), float(p[1])] for p in line.bbox]   # type: ignore
            text = str(getattr(line, "text", getattr(line, "rec_text", ""))).strip()
            conf = float(getattr(line, "score", getattr(line, "rec_score", 0.0)))  # type: ignore
            if text:
                results.append((pts, text, conf))
        except AttributeError:
            pass

    return results


def _worker_ocr(payload: bytes) -> tuple[list, float, int, int]:
    """
    Top-level worker function — pickle-safe, runs in a child process.

    Receives raw image bytes → preprocess → PaddleOCR → normalised result list.
    Returns (results, upscale_scale, orig_h, orig_w).
    """
    arr = np.frombuffer(payload, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return [], 1.0, 0, 0

    orig_h, orig_w = img.shape[:2]

    # ── Preprocess ──────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)

    scale  = 1.0
    target = 1_400
    if max(orig_h, orig_w) < target:
        scale = target / max(orig_h, orig_w)
        eq    = cv2.resize(eq, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    blur  = cv2.GaussianBlur(eq, (0, 0), 3)
    sharp = cv2.addWeighted(eq, 1.7, blur, -0.7, 0)
    proc  = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    # ── OCR ─────────────────────────────────────────────────────────────────
    raw     = _paddle.ocr(proc)  # type: ignore[union-attr]
    results = _parse_paddle_result(raw)

    return results, scale, orig_h, orig_w


# =============================================================================
#  MAIN-PROCESS CODE — FastAPI app, image helpers, diff logic
# =============================================================================

_pool: ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager — replaces the deprecated
    @app.on_event("startup") / @app.on_event("shutdown") pattern.
    """
    global _pool

    log.info("Pre-warming PaddleOCR models in main process…")
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
        _prewarmer = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )

        _prewarmer.ocr(np.zeros((10, 10, 3), np.uint8))
        del _prewarmer
        log.info("Models pre-downloaded successfully.")
    except Exception as e:
        log.warning("Pre-download failed (non-fatal): %s", e)

    # ── Startup ─────────────────────────────────────────────────────────────
    log.info("Starting 2 OCR worker processes (%d threads each)…", _THREADS_PER_WORKER)
    _pool = ProcessPoolExecutor(max_workers=2, initializer=_worker_init)

    
    loop  = asyncio.get_event_loop()
    blank = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))[1].tobytes()
    await asyncio.gather(
        loop.run_in_executor(_pool, _worker_ocr, blank),
        loop.run_in_executor(_pool, _worker_ocr, blank),
    )
    log.info("Both OCR workers ready.")

    yield  


    if _pool:
        _pool.shutdown(wait=False)


app = FastAPI(title="Label Diff Scanner", version="4.0", lifespan=lifespan)


def _load_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — is this a valid JPEG/PNG/WEBP?")
    return img


def _preprocess_for_preview(img: np.ndarray) -> np.ndarray:
    """CLAHE + USM for the browser preview PNG (runs in main process)."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    h, w  = eq.shape
    if max(h, w) < 1_400:
        sc = 1_400 / max(h, w)
        eq = cv2.resize(eq, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    blur  = cv2.GaussianBlur(eq, (0, 0), 3)
    sharp = cv2.addWeighted(eq, 1.7, blur, -0.7, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()



def _clean(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _split_token(pts: list, text: str, conf: float, scale: float,
                 orig_h: int, orig_w: int) -> list[dict]:
    """
    Tokenisation normalisation — split multi-word OCR tokens and distribute
    their bounding box proportionally by character count.
    """
    parts = [p for p in text.strip().split() if _clean(p)]
    if not parts:
        return []

    orig_pts = [[p[0] / scale, p[1] / scale] for p in pts]
    xs = [p[0] for p in orig_pts]
    ys = [p[1] for p in orig_pts]
    x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
    x1, y1 = min(float(orig_w), max(xs)), min(float(orig_h), max(ys))
    total_w = x1 - x0

    if len(parts) == 1:
        return [{
            "text":    parts[0],
            "conf":    round(conf, 3),
            "bbox":    {"x": round(x0), "y": round(y0),
                        "w": round(total_w), "h": round(y1 - y0)},
            "polygon": [[round(p[0]), round(p[1])] for p in orig_pts],
        }]

    total_chars = sum(len(p) for p in parts)
    result, cursor_x = [], x0
    for part in parts:
        pw       = total_w * len(part) / total_chars
        px0, px1 = cursor_x, cursor_x + pw
        result.append({
            "text":    part,
            "conf":    round(conf, 3),
            "bbox":    {"x": round(px0), "y": round(y0),
                        "w": round(pw),  "h": round(y1 - y0)},
            "polygon": [
                [round(px0), round(y0)], [round(px1), round(y0)],
                [round(px1), round(y1)], [round(px0), round(y1)],
            ],
        })
        cursor_x += pw
    return result


def _ocr_results_to_words(
    results: list[tuple[list, str, float]],
    scale: float,
    orig_h: int,
    orig_w: int,
) -> list[dict]:
    words: list[dict] = []
    for (pts, text, conf) in results:
        if conf < 0.25:
            continue
        words.extend(_split_token(pts, text, conf, scale, orig_h, orig_w))
    words.sort(key=lambda w: (w["bbox"]["y"] // 20, w["bbox"]["x"]))
    return words


def _levenshtein(a: str, b: str) -> int:
    if a == b:  return 0
    if not a:   return len(b)
    if not b:   return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    ca, cb = _clean(a), _clean(b)
    if not ca and not cb:  return 1.0
    if not ca or  not cb:  return 0.0
    return 1.0 - _levenshtein(ca, cb) / max(len(ca), len(cb))


_FUZZY_MATCH_THRESHOLD = 0.75


def _fuzzy_lcs_diff(wa: list[dict], wb: list[dict]) -> list[dict]:
    m, n = len(wa), len(wb)
    sim  = [[_similarity(wa[i]["text"], wb[j]["text"]) for j in range(n)]
            for i in range(m)]
    dp   = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sim[i-1][j-1] >= _FUZZY_MATCH_THRESHOLD:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    i, j, out = m, n, []
    while i > 0 or j > 0:
        if (i > 0 and j > 0
                and sim[i-1][j-1] >= _FUZZY_MATCH_THRESHOLD
                and dp[i][j] == dp[i-1][j-1] + 1):
            s = sim[i-1][j-1]
            out.append(
                {"type": "match",    "ref": wa[i-1], "user": wb[j-1]}
                if s >= 1.0 else
                {"type": "modified", "ref": wa[i-1], "user": wb[j-1],
                 "similarity": round(s, 3)}
            )
            i -= 1; j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            out.append({"type": "added",   "word": wb[j-1]}); j -= 1
        else:
            out.append({"type": "removed", "word": wa[i-1]}); i -= 1
    return list(reversed(out))


def _spatial_second_pass(diff: list[dict], ref_h: int, user_h: int) -> list[dict]:
    SPATIAL_TOL = 0.08

    def y_norm(word: dict, img_h: int) -> float:
        b = word["bbox"]
        return (b["y"] + b["h"] / 2) / img_h

    removed_idx = [i for i, d in enumerate(diff) if d["type"] == "removed"]
    added_idx   = [i for i, d in enumerate(diff) if d["type"] == "added"]
    used_added: set[int] = set()
    promotions: dict[int, dict | None] = {}

    for ri in removed_idx:
        rw = diff[ri]["word"]
        ry = y_norm(rw, ref_h)
        best_dist, best_ai = float("inf"), None
        for ai in added_idx:
            if ai in used_added:
                continue
            dist = abs(ry - y_norm(diff[ai]["word"], user_h))
            if dist < SPATIAL_TOL and dist < best_dist:
                best_dist, best_ai = dist, ai
        if best_ai is not None:
            s = _similarity(rw["text"], diff[best_ai]["word"]["text"])
            promotions[ri]      = {"type": "modified", "ref": rw,
                                    "user": diff[best_ai]["word"],
                                    "similarity": round(s, 3)}
            promotions[best_ai] = None
            used_added.add(best_ai)

    if not promotions:
        return diff
    result = []
    for i, d in enumerate(diff):
        if i not in promotions:
            result.append(d)
        elif promotions[i] is not None:
            result.append(promotions[i])
    return result


def _smart_diff(wa: list[dict], wb: list[dict],
                ref_h: int, user_h: int) -> list[dict]:
    return _spatial_second_pass(_fuzzy_lcs_diff(wa, wb), ref_h, user_h)



@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


@app.post("/compare")
async def compare(
    ref_image:  UploadFile = File(...),
    user_image: UploadFile = File(...),
) -> dict:
    if _pool is None:
        raise HTTPException(503, "OCR workers are still starting — please retry.")

    ref_bytes, user_bytes = await asyncio.gather(
        ref_image.read(), user_image.read()
    )

    try:
        ref_raw  = _load_image(ref_bytes)
        user_raw = _load_image(user_bytes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    ref_h,  ref_w  = ref_raw.shape[:2]
    user_h, user_w = user_raw.shape[:2]

    log.info("Dispatching both images to OCR workers in parallel…")
    t0   = asyncio.get_event_loop().time()
    loop = asyncio.get_event_loop()

    (ref_results,  ref_scale,  _, _), \
    (user_results, user_scale, _, _) = await asyncio.gather(
        loop.run_in_executor(_pool, _worker_ocr, ref_bytes),
        loop.run_in_executor(_pool, _worker_ocr, user_bytes),
    )

    elapsed = asyncio.get_event_loop().time() - t0
    log.info("Parallel OCR complete in %.2f s", elapsed)

    ref_words  = _ocr_results_to_words(ref_results,  ref_scale,  ref_h,  ref_w)
    user_words = _ocr_results_to_words(user_results, user_scale, user_h, user_w)
    log.info("ref: %d words  user: %d words", len(ref_words), len(user_words))

    diff     = _smart_diff(ref_words, user_words, ref_h, user_h)
    removed  = sum(1 for d in diff if d["type"] == "removed")
    added    = sum(1 for d in diff if d["type"] == "added")
    modified = sum(1 for d in diff if d["type"] == "modified")
    log.info("Diff — %d modified  %d removed  %d added", modified, removed, added)

    return {
        "diff":            diff,
        "ref_word_count":  len(ref_words),
        "user_word_count": len(user_words),
        "removed_count":   removed,
        "added_count":     added,
        "modified_count":  modified,
        "ocr_time_s":      round(elapsed, 2),
        "ref_size":        {"w": int(ref_w),  "h": int(ref_h)},
        "user_size":       {"w": int(user_w), "h": int(user_h)},
        "ref_preview":     _to_b64(_preprocess_for_preview(ref_raw)),
        "user_preview":    _to_b64(_preprocess_for_preview(user_raw)),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "workers_ready": _pool is not None}


if __name__ == "__main__":
    import multiprocessing
    import uvicorn

    multiprocessing.set_start_method("spawn", force=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)