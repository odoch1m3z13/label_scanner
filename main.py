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
    """
    global _paddle

    # Tell PaddlePaddle how many threads this worker may use
    os.environ["OMP_NUM_THREADS"]    = str(_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"]    = str(_THREADS_PER_WORKER)
    os.environ["PADDLE_NUM_THREADS"] = str(_THREADS_PER_WORKER)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    from paddleocr import PaddleOCR  # noqa: PLC0415

    # PaddleOCR 2.7.x — full kwargs available, stable on CPU
    _paddle = PaddleOCR(
        use_angle_cls=False,  # FIX: Disabled for 300ms+ speedup per scan
        lang="en",
        use_gpu=False,
        cpu_threads=_THREADS_PER_WORKER,
        show_log=False,
        enable_mkldnn=True,   # Intel MKL-DNN — ~30% faster on x86 CPUs
    )

    logging.getLogger(__name__).info(
        "Worker PID %d: PaddleOCR ready (%d threads)", os.getpid(), _THREADS_PER_WORKER
    )


def _parse_paddle_result(raw: object) -> list[tuple[list, str, float]]:
    """Parse PaddleOCR result defensively across 2.x and 3.x return formats."""
    results: list[tuple[list, str, float]] = []

    if raw is None:
        return results

    try:
        lines = raw[0]  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        lines = raw

    if lines is None:
        return results

    for line in lines:
        if line is None:
            continue

        try:
            pts_raw, text_conf = line
            text, conf = text_conf
            pts = [[float(p[0]), float(p[1])] for p in pts_raw]
            results.append((pts, str(text).strip(), float(conf)))
            continue
        except (TypeError, ValueError, IndexError):
            pass

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
    """
    arr = np.frombuffer(payload, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return [], 1.0, 0, 0

    orig_h, orig_w = img.shape[:2]

    # ── Preprocess ──────────────────────────────────────────────────────────
    # FIX: Scale DOWN before applying heavy OpenCV math to save massive CPU cycles
    scale  = 1.0
    target = 1_400
    max_dim = max(orig_h, orig_w)
    if max_dim != target:
        scale = target / max_dim
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    
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
    global _pool
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
        raise ValueError("Could not decode image.")
    return img


def _preprocess_for_preview(img: np.ndarray) -> np.ndarray:
    # Scale first, then filter (faster)
    h, w = img.shape[:2]
    if max(h, w) > 1_400:
        sc = 1_400 / max(h, w)
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (0, 0), 3)
    sharp = cv2.addWeighted(eq, 1.7, blur, -0.7, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _to_b64(img: np.ndarray) -> str:
    # FIX: Use JPEG instead of PNG for 5x faster network transmission
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def _clean(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _split_token(pts: list, text: str, conf: float, scale: float,
                 orig_h: int, orig_w: int) -> list[dict]:
                 
    # FIX: Pad symbols so "OZ(640g)" splits correctly to "OZ" and "640g"
    spaced_text = re.sub(r'([^\w\s])', r' \1 ', text)
    parts = [p for p in spaced_text.strip().split() if _clean(p)]
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
        
    if not words:
        return []

    # FIX: Dynamic line height. Fixes the massive "MillDCast & Thyme" box
    median_h = max(10.0, float(np.median([w["bbox"]["h"] for w in words])))
    words.sort(key=lambda w: (int(w["bbox"]["y"] / (median_h * 0.5)), w["bbox"]["x"]))
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


_COLOR_DELTA_THRESHOLD = 15.0  
_MIN_INK_PIXELS        = 8     

def _sample_ink_lab(img_bgr: np.ndarray, polygon: list) -> tuple[float, float] | None:
    h, w = img_bgr.shape[:2]
    pts    = np.array(polygon, dtype=np.int32)
    rx, ry, rw, rh = cv2.boundingRect(pts)
    rx  = max(0, rx);  ry  = max(0, ry)
    rw  = min(rw, w - rx);  rh  = min(rh, h - ry)
    if rw < 2 or rh < 2:
        return None

    crop = img_bgr[ry:ry+rh, rx:rx+rw].copy()
    shifted = pts - np.array([rx, ry])
    poly_mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [shifted], 255)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_mask = cv2.bitwise_and(poly_mask, otsu)
    ink_count = int(np.count_nonzero(ink_mask))
    if ink_count < _MIN_INK_PIXELS:
        return None

    lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    means = cv2.mean(lab, mask=ink_mask)   
    return float(means[1]), float(means[2])   


def _compute_chroma_calibration(
    diff: list[dict],
    ref_raw: np.ndarray,
    user_raw: np.ndarray,
) -> tuple[float, float]:
    a_deltas: list[float] = []
    b_deltas: list[float] = []
    for d in diff:
        if d["type"] != "match":
            continue
        rc = _sample_ink_lab(ref_raw,  d["ref"]["polygon"])
        uc = _sample_ink_lab(user_raw, d["user"]["polygon"])
        if rc and uc:
            a_deltas.append(rc[0] - uc[0])
            b_deltas.append(rc[1] - uc[1])
    if not a_deltas:
        return 0.0, 0.0
    return float(np.median(a_deltas)), float(np.median(b_deltas))


def _detect_text_color_changes(
    diff: list[dict],
    ref_raw: np.ndarray,
    user_raw: np.ndarray,
) -> tuple[list[dict], tuple[float, float]]:
    a_off, b_off = _compute_chroma_calibration(diff, ref_raw, user_raw)
    result: list[dict] = []
    for d in diff:
        if d["type"] == "match":
            rc = _sample_ink_lab(ref_raw,  d["ref"]["polygon"])
            uc = _sample_ink_lab(user_raw, d["user"]["polygon"])
            if rc and uc:
                da    = (rc[0] - uc[0]) - a_off
                db    = (rc[1] - uc[1]) - b_off
                delta = (da**2 + db**2) ** 0.5
                if delta > _COLOR_DELTA_THRESHOLD:
                    result.append({**d, "type": "color_changed",
                                   "color_delta": round(delta, 1)})
                    continue
        result.append(d)
    return result, (a_off, b_off)


_MIN_TAMPER_AREA_FRAC = 0.002   
_TAMPER_COLOR_THRESH  = 18.0    
_MIN_KEYPOINTS        = 12      


def _detect_visual_tampering(
    ref_raw: np.ndarray,
    user_raw: np.ndarray,
    flagged_ref_boxes: list[dict],
    chroma_offset: tuple[float, float],
) -> tuple[list[dict], list[dict], str]:

    ref_gray  = cv2.cvtColor(ref_raw,  cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_raw, cv2.COLOR_BGR2GRAY)

    orb      = cv2.ORB_create(nfeatures=3000)
    kp1, d1  = orb.detectAndCompute(ref_gray,  None)
    kp2, d2  = orb.detectAndCompute(user_gray, None)

    if d1 is None or d2 is None or len(kp1) < _MIN_KEYPOINTS or len(kp2) < _MIN_KEYPOINTS:
        return [], [], "insufficient_keypoints"

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)
    good    = matches[: min(200, len(matches))]
    if len(good) < _MIN_KEYPOINTS:
        return [], [], "insufficient_matches"

    src_pts  = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts  = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, hmask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or hmask is None or int(hmask.ravel().sum()) < _MIN_KEYPOINTS:
        return [], [], "homography_failed"

    ref_h, ref_w   = ref_raw.shape[:2]
    user_h, user_w = user_raw.shape[:2]
    user_aligned   = cv2.warpPerspective(user_raw, H, (ref_w, ref_h))

    valid = (user_aligned[:, :, 0] > 0) | (user_aligned[:, :, 1] > 0) | (user_aligned[:, :, 2] > 0)

    ref_lab  = cv2.cvtColor(ref_raw,      cv2.COLOR_BGR2LAB).astype(np.float32)
    user_lab = cv2.cvtColor(user_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)

    a_off, b_off = chroma_offset
    diff_a = (ref_lab[:, :, 1] - user_lab[:, :, 1]) - a_off
    diff_b = (ref_lab[:, :, 2] - user_lab[:, :, 2]) - b_off
    chroma_diff          = np.sqrt(diff_a**2 + diff_b**2)
    chroma_diff[~valid]  = 0.0

    binary = (chroma_diff > _TAMPER_COLOR_THRESH).astype(np.uint8) * 255
    kclose = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    kopen  = cv2.getStructuringElement(cv2.MORPH_RECT,  (5,  5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kclose)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kopen)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area  = ref_h * ref_w * _MIN_TAMPER_AREA_FRAC
    max_area  = ref_h * ref_w * 0.08  # FIX: Ignore areas > 8% (eliminates giant glare boxes)

    ref_boxes:  list[dict] = []
    user_boxes: list[dict] = []

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # FIX: Check against both min and max area thresholds
        if area < min_area or area > max_area:
            continue
            
        x, y, bw, bh = cv2.boundingRect(cnt)

        dominated = False
        for tb in flagged_ref_boxes:
            tx, ty, tw, th = tb["x"], tb["y"], tb["w"], tb["h"]
            ix = max(0, min(x + bw, tx + tw) - max(x, tx))
            iy = max(0, min(y + bh, ty + th) - max(y, ty))
            if bw * bh > 0 and (ix * iy) / (bw * bh) > 0.5:
                dominated = True
                break
        if dominated:
            continue

        ref_boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)})

        if H_inv is not None:
            corners = np.float32([
                [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
            ]).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
            uxs = np.clip(mapped[:, 0], 0, user_w)
            uys = np.clip(mapped[:, 1], 0, user_h)
            user_boxes.append({
                "x": int(uxs.min()), "y": int(uys.min()),
                "w": int(uxs.max() - uxs.min()),
                "h": int(uys.max() - uys.min()),
            })

    return ref_boxes, user_boxes, "ok"


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

    diff             = _smart_diff(ref_words, user_words, ref_h, user_h)
    diff, chroma_off = _detect_text_color_changes(diff, ref_raw, user_raw)

    removed       = sum(1 for d in diff if d["type"] == "removed")
    added         = sum(1 for d in diff if d["type"] == "added")
    modified      = sum(1 for d in diff if d["type"] == "modified")
    color_changed = sum(1 for d in diff if d["type"] == "color_changed")

    flagged_ref_boxes: list[dict] = []
    for d in diff:
        if d["type"] == "removed":
            flagged_ref_boxes.append(d["word"]["bbox"])
        elif d["type"] in ("modified", "color_changed"):
            flagged_ref_boxes.append(d["ref"]["bbox"])

    tamper_ref, tamper_user, tamper_status = _detect_visual_tampering(
        ref_raw, user_raw, flagged_ref_boxes, chroma_off
    )

    log.info(
        "Diff — %d modified  %d removed  %d added  %d color_changed  "
        "%d tamper_regions  (tamper_status=%s)",
        modified, removed, added, color_changed, len(tamper_ref), tamper_status,
    )

    return {
        "diff":                diff,
        "ref_word_count":      len(ref_words),
        "user_word_count":     len(user_words),
        "removed_count":       removed,
        "added_count":         added,
        "modified_count":      modified,
        "color_changed_count": color_changed,
        "tamper_boxes_ref":    tamper_ref,
        "tamper_boxes_user":   tamper_user,
        "tamper_status":       tamper_status,
        "ocr_time_s":          round(elapsed, 2),
        "ref_size":            {"w": int(ref_w),  "h": int(ref_h)},
        "user_size":           {"w": int(user_w), "h": int(user_h)},
        "ref_preview":         _to_b64(_preprocess_for_preview(ref_raw)),
        "user_preview":        _to_b64(_preprocess_for_preview(user_raw)),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "workers_ready": _pool is not None}


if __name__ == "__main__":
    import multiprocessing
    import uvicorn
    # "spawn" avoids fork-safety issues with Paddle's C++ runtime
    multiprocessing.set_start_method("spawn", force=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)