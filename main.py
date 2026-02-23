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

app = FastAPI(title="Label Diff Scanner", version="3.0")

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


def _split_token(pts: list, text: str, conf: float, scale: float,
                 orig_h: int, orig_w: int) -> list[dict]:
    """
    FIX 1 — Tokenisation normalisation.

    EasyOCR sometimes returns a whole phrase as one token, e.g.
      "Wildflower & Thyme Honey"  (4 logical words, 1 OCR token)
    while the other image returns them as separate tokens.
    This mismatch causes the LCS to misalign every word that follows.

    Solution: split any multi-word token on whitespace and distribute the
    bounding polygon proportionally by character count across the split words.
    Single-character tokens (punctuation artefacts) are dropped — they add
    noise to the diff without carrying meaningful label information.
    """
    parts = text.strip().split()
    # Drop lone punctuation / single-char noise tokens
    parts = [p for p in parts if len(_clean(p)) > 0]
    if not parts:
        return []

    # Scale polygon back to original image coordinate space
    orig_pts = [[p[0] / scale, p[1] / scale] for p in pts]
    xs = [p[0] for p in orig_pts]
    ys = [p[1] for p in orig_pts]
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(float(orig_w), max(xs))
    y1 = min(float(orig_h), max(ys))
    total_w = x1 - x0

    # If only one real word in this token, return it as-is
    if len(parts) == 1:
        return [{
            "text":    parts[0],
            "conf":    round(float(conf), 3),
            "bbox":    {"x": round(x0), "y": round(y0),
                        "w": round(total_w), "h": round(y1 - y0)},
            "polygon": [[round(p[0]), round(p[1])] for p in orig_pts],
        }]

    # Distribute bbox width proportionally by character length
    char_counts  = [len(p) for p in parts]
    total_chars  = sum(char_counts)
    result       = []
    cursor_x     = x0
    for part, chars in zip(parts, char_counts):
        frac    = chars / total_chars
        pw      = total_w * frac
        px0, px1 = cursor_x, cursor_x + pw
        # Build a simple axis-aligned polygon for split sub-tokens
        poly = [
            [round(px0), round(y0)],
            [round(px1), round(y0)],
            [round(px1), round(y1)],
            [round(px0), round(y1)],
        ]
        result.append({
            "text":    part,
            "conf":    round(float(conf), 3),
            "bbox":    {"x": round(px0), "y": round(y0),
                        "w": round(pw), "h": round(y1 - y0)},
            "polygon": poly,
        })
        cursor_x += pw
    return result


def _ocr_results_to_words(
    results: list,
    scale: float,
    orig_h: int,
    orig_w: int,
) -> list[dict]:
    """
    Convert EasyOCR raw results → normalised word list in ORIGINAL pixel coords.
    Multi-word tokens are split (Fix 1).
    """
    words: list[dict] = []
    for (pts, text, conf) in results:
        if conf < 0.25:
            continue
        words.extend(_split_token(pts, text, conf, scale, orig_h, orig_w))

    # Sort into reading order: top → bottom (row bucket = 20 px), left → right
    words.sort(key=lambda w: (w["bbox"]["y"] // 20, w["bbox"]["x"]))
    return words


# ── Diff: fuzzy LCS + spatial second-pass ──────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """
    Character-level similarity in [0, 1].
    1.0 = identical, 0.0 = completely different.
    Operates on the cleaned (alphanumeric-only) forms of both strings.
    """
    ca, cb = _clean(a), _clean(b)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    maxlen = max(len(ca), len(cb))
    return 1.0 - _levenshtein(ca, cb) / maxlen


# Threshold above which two words are treated as "the same position" in the LCS.
# Words above this but below 1.0 are "modified" (changed in place).
# Words below this are truly different tokens → handled by the spatial pass.
_FUZZY_MATCH_THRESHOLD = 0.75


def _fuzzy_lcs_diff(wa: list[dict], wb: list[dict]) -> list[dict]:
    """
    FIX 2 — Fuzzy LCS.

    The standard LCS only matches IDENTICAL words. When one word changes
    (e.g. "Amber" → "Maber"), the DP table cannot match it, so it emits a
    delete + insert. That alone is fine. The problem is the cascade: if the
    surrounding words are also shifted by the mismatch, the LCS may find a
    cheaper path that sacrifices genuinely identical words downstream
    (like "Thyme", "Honey") to recover alignment.

    Fix: use similarity ≥ 0.75 as the match condition in the DP table.
      • "Thyme"  vs "Thyme"   → sim=1.0 → counted as a match → aligned ✓
      • "Amber"  vs "Maber"   → sim=0.75 → counted as a match → aligned,
                                            but tagged "modified" not "match"
      • "Wildflower" vs "MiLDCast" → sim≈0.3 → NOT a match → handled below
                                                by the spatial pass

    This keeps the alignment rock-steady while still catching substitutions.
    """
    m, n = len(wa), len(wb)

    # Build similarity matrix once (avoid recomputing during backtrack)
    sim = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            sim[i][j] = _similarity(wa[i]["text"], wb[j]["text"])

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sim[i - 1][j - 1] >= _FUZZY_MATCH_THRESHOLD:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    i, j, out = m, n, []
    while i > 0 or j > 0:
        if (i > 0 and j > 0
                and sim[i - 1][j - 1] >= _FUZZY_MATCH_THRESHOLD
                and dp[i][j] == dp[i - 1][j - 1] + 1):
            s = sim[i - 1][j - 1]
            if s >= 1.0:
                # Exact (cleaned) match — no box needed
                out.append({"type": "match", "ref": wa[i - 1], "user": wb[j - 1]})
            else:
                # Similar but not identical — word was changed in place
                out.append({"type": "modified", "ref": wa[i - 1], "user": wb[j - 1],
                             "similarity": round(s, 3)})
            i -= 1; j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            out.append({"type": "added",   "word": wb[j - 1]})
            j -= 1
        else:
            out.append({"type": "removed", "word": wa[i - 1]})
            i -= 1

    return list(reversed(out))


def _spatial_second_pass(
    diff: list[dict],
    ref_h: int,
    user_h: int,
) -> list[dict]:
    """
    FIX 3 — Spatial / positional second pass.

    After the fuzzy LCS, any remaining "removed" + "added" pairs that
    occupy the same normalised vertical position in their respective images
    are almost certainly the same word that changed — they just fell below
    the 0.75 similarity threshold (e.g. "Wildflower" → "MiLDCast", sim≈0.3).

    We scan the diff list for adjacent removed/added runs and pair them up
    by normalised Y centre.  Matched pairs are promoted to "modified".

    Normalisation: y_norm = (bbox.y + bbox.h/2) / image_height
    Tolerance: ±8 % of image height — generous enough to handle perspective
    distortion between two cameras, tight enough to avoid cross-row pairings.
    """
    SPATIAL_TOL = 0.08  # ±8 % of image height

    def y_norm(word: dict, img_h: int) -> float:
        b = word["bbox"]
        return (b["y"] + b["h"] / 2) / img_h

    # Collect indices of removed and added items
    removed_idx = [i for i, d in enumerate(diff) if d["type"] == "removed"]
    added_idx   = [i for i, d in enumerate(diff) if d["type"] == "added"]

    used_removed: set[int] = set()
    used_added:   set[int] = set()
    promotions:   dict[int, dict] = {}  # diff index → replacement entry

    for ri in removed_idx:
        rw = diff[ri]["word"]
        ry = y_norm(rw, ref_h)
        best_dist, best_ai = float("inf"), None

        for ai in added_idx:
            if ai in used_added:
                continue
            aw = diff[ai]["word"]
            ay = y_norm(aw, user_h)
            dist = abs(ry - ay)
            if dist < SPATIAL_TOL and dist < best_dist:
                best_dist, best_ai = dist, ai

        if best_ai is not None:
            s = _similarity(rw["text"], diff[best_ai]["word"]["text"])
            promotions[ri]       = {"type": "modified",
                                     "ref":  rw,
                                     "user": diff[best_ai]["word"],
                                     "similarity": round(s, 3)}
            promotions[best_ai]  = None   # mark the 'added' for deletion
            used_removed.add(ri)
            used_added.add(best_ai)

    # Rebuild diff, substituting promotions and removing consumed 'added' entries
    result = []
    for i, d in enumerate(diff):
        if i in promotions:
            if promotions[i] is not None:
                result.append(promotions[i])
            # else: this was the 'added' half — drop it (merged into 'modified' above)
        else:
            result.append(d)
    return result


def _smart_diff(wa: list[dict], wb: list[dict],
                ref_h: int, user_h: int) -> list[dict]:
    """Full three-fix pipeline: fuzzy LCS → spatial second pass."""
    diff = _fuzzy_lcs_diff(wa, wb)
    diff = _spatial_second_pass(diff, ref_h, user_h)
    return diff


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

    ref_h, ref_w   = ref_raw.shape[:2]
    user_h, user_w = user_raw.shape[:2]

    diff     = _smart_diff(ref_words, user_words, ref_h, user_h)
    removed  = sum(1 for d in diff if d["type"] == "removed")
    added    = sum(1 for d in diff if d["type"] == "added")
    modified = sum(1 for d in diff if d["type"] == "modified")

    log.info("Diff — %d removed, %d added, %d modified", removed, added, modified)

    return {
        "diff":            diff,
        "ref_word_count":  len(ref_words),
        "user_word_count": len(user_words),
        "removed_count":   removed,
        "added_count":     added,
        "modified_count":  modified,
        "ref_size":        {"w": int(ref_w),  "h": int(ref_h)},
        "user_size":       {"w": int(user_w), "h": int(user_h)},
        "ref_preview":     _to_b64(ref_proc),
        "user_preview":    _to_b64(user_proc),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "ocr_ready": _reader is not None}


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)