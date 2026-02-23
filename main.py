"""
Label Diff Scanner — FastAPI + EasyOCR + OpenCV backend
Run:  python main.py
Open: http://localhost:8000
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Label Diff Scanner", version="2.0")

# ── EasyOCR reader — loaded once at startup (first run downloads ~400 MB) ──────
_reader: easyocr.Reader | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _reader
    log.info("Loading EasyOCR model — first run downloads ~400 MB, subsequent runs are instant…")
    loop = asyncio.get_event_loop()
    _reader = await loop.run_in_executor(
        None,
        lambda: easyocr.Reader(["en"], gpu=False, verbose=False),
    )
    log.info("✓  EasyOCR ready.")


# ── Image helpers ───────────────────────────────────────────────────────────────

def _load_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — is this a valid JPEG/PNG/WEBP?")
    return img


def _preprocess(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Returns (processed_BGR_image, upscale_factor).

    Pipeline:
      1. Grayscale conversion
      2. CLAHE  (Contrast Limited Adaptive Histogram Equalisation)
         — removes per-tile lighting differences; compensates for glare, shadows,
           and the different exposures produced by two different cameras.
      3. Upscale to ≥1 400 px on the long edge
         — tiny product label text becomes readable.
      4. Unsharp Masking (USM)
         — sharpens edges without amplifying sensor noise (better than Laplacian).
      5. Convert back to BGR so EasyOCR and the preview encoder work correctly.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE: divide image into 8×8 tiles, equalize each, interpolate borders
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    # Upscale small images
    h, w = eq.shape
    scale = 1.0
    target = 1_400
    if max(h, w) < target:
        scale = target / max(h, w)
        eq = cv2.resize(eq, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Unsharp masking: sharp = original + amount * (original − gaussian_blur)
    blur  = cv2.GaussianBlur(eq, (0, 0), 3)
    sharp = cv2.addWeighted(eq, 1.7, blur, -0.7, 0)

    bgr = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return bgr, scale


def _to_b64(img: np.ndarray) -> str:
    """Encode an OpenCV image as a base64 PNG string."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


# ── OCR helpers ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalise a word for comparison: lowercase, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _ocr_results_to_words(
    results: list,
    scale: float,
    orig_h: int,
    orig_w: int,
) -> list[dict]:
    """
    Convert EasyOCR raw results → normalised word list in ORIGINAL pixel coords.

    EasyOCR returns:  [ ([[x0,y0],[x1,y1],[x2,y2],[x3,y3]], text, confidence), … ]
    The four corners may be rotated (handles slanted / curved text naturally).

    We store both:
      • bbox    — axis-aligned box for simple display
      • polygon — actual four-corner polygon for accurate rotated-box drawing
    """
    words: list[dict] = []

    for (pts, text, conf) in results:
        text = text.strip()
        if not _clean(text) or conf < 0.25:
            continue

        # Scale back to original image coordinate space
        orig_pts = [[p[0] / scale, p[1] / scale] for p in pts]
        xs = [p[0] for p in orig_pts]
        ys = [p[1] for p in orig_pts]

        x0 = max(0.0, min(xs))
        y0 = max(0.0, min(ys))
        x1 = min(float(orig_w), max(xs))
        y1 = min(float(orig_h), max(ys))

        words.append(
            {
                "text":    text,
                "conf":    round(float(conf), 3),
                "bbox":    {
                    "x": round(x0), "y": round(y0),
                    "w": round(x1 - x0), "h": round(y1 - y0),
                },
                # 4-corner polygon — used by frontend for rotated box drawing
                "polygon": [[round(p[0]), round(p[1])] for p in orig_pts],
            }
        )

    # Sort into reading order: top → bottom (20 px row tolerance), left → right
    words.sort(key=lambda w: (w["bbox"]["y"] // 20, w["bbox"]["x"]))
    return words


# ── Diff: LCS-based word comparison ─────────────────────────────────────────────

def _lcs_diff(wa: list[dict], wb: list[dict]) -> list[dict]:
    """
    Longest Common Subsequence diff on two word lists.
    Returns a list of items, each with type ∈ { 'match', 'removed', 'added' }.
      removed → word present in original, missing from scan  (draw green on ref)
      added   → word present in scan, not in original        (draw red on user)
    """
    m, n = len(wa), len(wb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if _clean(wa[i - 1]["text"]) == _clean(wb[j - 1]["text"]):
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i, j, out = m, n, []
    while i > 0 or j > 0:
        if (
            i > 0 and j > 0
            and _clean(wa[i - 1]["text"]) == _clean(wb[j - 1]["text"])
        ):
            out.append({"type": "match", "ref": wa[i - 1], "user": wb[j - 1]})
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            out.append({"type": "added", "word": wb[j - 1]})
            j -= 1
        else:
            out.append({"type": "removed", "word": wa[i - 1]})
            i -= 1

    return list(reversed(out))


# ── API ─────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/compare")
async def compare(
    ref_image:  UploadFile = File(..., description="Original / reference label"),
    user_image: UploadFile = File(..., description="Scanned label to verify"),
) -> dict:
    if _reader is None:
        raise HTTPException(503, "OCR engine is still loading — please wait a moment and retry.")

    # Read both uploads concurrently
    ref_bytes, user_bytes = await asyncio.gather(
        ref_image.read(),
        user_image.read(),
    )

    try:
        ref_raw  = _load_image(ref_bytes)
        user_raw = _load_image(user_bytes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Preprocess
    ref_proc,  ref_scale  = _preprocess(ref_raw)
    user_proc, user_scale = _preprocess(user_raw)

    # OCR — run in thread executor so the async event loop stays unblocked.
    # EasyOCR's reader is NOT thread-safe so we run the two calls sequentially.
    loop = asyncio.get_event_loop()

    log.info("Running OCR on reference image…")
    ref_results = await loop.run_in_executor(
        None,
        lambda: _reader.readtext(ref_proc, detail=1, paragraph=False),
    )

    log.info("Running OCR on user image…")
    user_results = await loop.run_in_executor(
        None,
        lambda: _reader.readtext(user_proc, detail=1, paragraph=False),
    )

    ref_words  = _ocr_results_to_words(ref_results,  ref_scale,  *ref_raw.shape[:2])
    user_words = _ocr_results_to_words(user_results, user_scale, *user_raw.shape[:2])

    log.info(
        "OCR complete — ref: %d words, user: %d words",
        len(ref_words), len(user_words),
    )

    diff = _lcs_diff(ref_words, user_words)
    removed = sum(1 for d in diff if d["type"] == "removed")
    added   = sum(1 for d in diff if d["type"] == "added")
    log.info("Diff — %d removed, %d added", removed, added)

    return {
        "diff":           diff,
        "ref_word_count": len(ref_words),
        "user_word_count": len(user_words),
        "removed_count":  removed,
        "added_count":    added,
        "ref_size":       {"w": int(ref_raw.shape[1]),  "h": int(ref_raw.shape[0])},
        "user_size":      {"w": int(user_raw.shape[1]), "h": int(user_raw.shape[0])},
        "ref_preview":    _to_b64(ref_proc),
        "user_preview":   _to_b64(user_proc),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ocr_ready": _reader is not None}


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)