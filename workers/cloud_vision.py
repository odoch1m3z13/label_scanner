"""
workers/cloud_vision.py — Async Google Cloud Vision client.

Responsibilities:
  - Send an image to Cloud Vision DOCUMENT_TEXT_DETECTION.
  - Parse the full document hierarchy (page → block → paragraph → word).
  - Infer LayoutRegions from block types returned by Cloud Vision.
  - Return a normalised SemanticMap used identically by Tier 1 and Tier 2.

Design notes:
  - Pure async — no ProcessPoolExecutor needed (network I/O releases the GIL).
  - Client is a lazily-initialised module-level singleton (one per process).
  - Used for BOTH reference registration and scan-time extraction to guarantee
    identical tokenisation — eliminates the "100g" vs "100"+"g" split problem.
  - Raw Cloud Vision JSON is preserved in SemanticMap.raw_response so the
    reference map can be reprocessed in future without another API call.

Credentials:
  Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
  or use Application Default Credentials (gcloud auth application-default login).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import cv2
import numpy as np
from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
from google.protobuf.json_format import MessageToDict

from config import (
    CLOUD_VISION_MAX_BYTES,
    CLOUD_VISION_MIN_CONFIDENCE,
)
from models.schemas import BoundingBox, LayoutRegion, Polygon, RegionType, SemanticMap, WordEntry

log = logging.getLogger(__name__)

# ── Client singleton ──────────────────────────────────────────────────────────
# Initialised once on first call to _get_client().
# The Vision client is thread-safe and can be reused across requests.
_cv_client: vision.ImageAnnotatorClient | None = None


def _get_client() -> vision.ImageAnnotatorClient:
    """
    Return (or lazily initialise) the Cloud Vision ImageAnnotatorClient.

    The client reads credentials from:
      1. GOOGLE_APPLICATION_CREDENTIALS env var (service account JSON path).
      2. Application Default Credentials (gcloud auth application-default login).

    Raises google.auth.exceptions.DefaultCredentialsError if neither is set.
    """
    global _cv_client
    if _cv_client is None:
        _cv_client = vision.ImageAnnotatorClient()
        log.info("Cloud Vision client initialised.")
    return _cv_client


# ── Vertex / geometry helpers ─────────────────────────────────────────────────

def _bbox_from_vertices(vertices: list) -> BoundingBox:
    """
    Convert a Cloud Vision vertex list to an axis-aligned BoundingBox.

    Cloud Vision returns vertices as a list of objects with optional .x / .y
    attributes. Missing coordinates default to 0 (Cloud Vision omits x or y
    when they are 0, so we must handle the absence gracefully).

    Vertex order: [top-left, top-right, bottom-right, bottom-left].
    We take min/max rather than assuming the order is exact — rotated text
    can produce non-standard orderings.
    """
    xs = [getattr(v, "x", 0) or 0 for v in vertices]
    ys = [getattr(v, "y", 0) or 0 for v in vertices]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    return BoundingBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))


def _polygon_from_vertices(vertices: list) -> Polygon:
    """
    Convert a Cloud Vision vertex list to a Polygon (ordered corner points).
    Missing x/y coordinates default to 0.
    """
    points = [
        (getattr(v, "x", 0) or 0, getattr(v, "y", 0) or 0)
        for v in vertices
    ]
    return Polygon(points=points)


# ── Block type → RegionType mapping ──────────────────────────────────────────
# Cloud Vision block types:
#   UNKNOWN=0, TEXT=1, TABLE=2, PICTURE=3, RULER=4, BARCODE=5
_BLOCK_TYPE_MAP: dict[int, RegionType] = {
    0: RegionType.unknown,
    1: RegionType.text,
    2: RegionType.table,
    3: RegionType.figure,    # PICTURE → figure
    4: RegionType.unknown,   # RULER — treat as unknown
    5: RegionType.barcode,
}


def _region_label_hint(block_type: int, bbox: BoundingBox, img_w: int, img_h: int) -> str | None:
    """
    Produce a human-readable label hint for a LayoutRegion based on block type
    and rough position heuristics.

    This is advisory only — used for logging and frontend display.
    The diff engine routes by RegionType, not by label.
    """
    if block_type == 5:
        return "barcode"
    if block_type == 3:
        # Rough position hint for figures
        cx = bbox.x + bbox.w / 2
        cy = bbox.y + bbox.h / 2
        if cx < img_w * 0.35:
            return "figure_left"
        if cx > img_w * 0.65:
            return "figure_right"
        return "figure_center"
    return None


# ── Core parser ───────────────────────────────────────────────────────────────

def _parse_document_text_annotation(
    response: AnnotateImageResponse,
    label_id: str,
    img_w:    int,
    img_h:    int,
) -> SemanticMap:
    """
    Parse a Cloud Vision DOCUMENT_TEXT_DETECTION response into a SemanticMap.

    Cloud Vision document hierarchy traversed:
      full_text_annotation
        .pages[0]
          .blocks[]          → LayoutRegion  (one per block)
            .paragraphs[]
              .words[]       → WordEntry     (one per word)
                .symbols[]   → joined into word text

    Word confidence:
      Cloud Vision reports confidence at the symbol level. We take the mean
      of all symbol confidences as the word confidence. If no symbol
      confidences are present we fall back to the paragraph confidence.

    Words below CLOUD_VISION_MIN_CONFIDENCE are excluded from the word list
      but their bounding boxes are still included in the region mask so the
      visual diff engine correctly ignores low-confidence text areas.

    Error handling:
      - Missing bounding boxes → skip the word, log a warning.
      - Empty pages / no annotation → return SemanticMap with empty lists.
      - API error in response → raise RuntimeError with the error message.
    """
    # ── Check for API-level errors ────────────────────────────────────────────
    if response.error.message:
        raise RuntimeError(
            f"Cloud Vision API error for label '{label_id}': {response.error.message}"
        )

    annotation = response.full_text_annotation
    if not annotation or not annotation.pages:
        log.warning("Cloud Vision returned no text annotation for label '%s'.", label_id)
        return SemanticMap(
            label_id=label_id,
            image_width=img_w,
            image_height=img_h,
            words=[],
            regions=[],
            raw_response=MessageToDict(response._pb),
        )

    page = annotation.pages[0]
    words:   list[WordEntry]    = []
    regions: list[LayoutRegion] = []

    for block_id, block in enumerate(page.blocks):
        # ── LayoutRegion from block ───────────────────────────────────────────
        block_type = block.block_type  # int enum value
        region_type = _BLOCK_TYPE_MAP.get(block_type, RegionType.unknown)

        if block.bounding_box and block.bounding_box.vertices:
            block_bbox  = _bbox_from_vertices(block.bounding_box.vertices)
            label_hint  = _region_label_hint(block_type, block_bbox, img_w, img_h)
            regions.append(LayoutRegion(
                region_type=region_type,
                bbox=block_bbox,
                label=label_hint,
            ))
        else:
            log.debug("Block %d has no bounding box — skipping region.", block_id)
            block_bbox = None

        # ── Words inside this block ───────────────────────────────────────────
        # Only TEXT blocks contribute to the word list.
        # Other block types (PICTURE, BARCODE, TABLE) have no word content.
        if region_type != RegionType.text:
            continue

        for para_id, para in enumerate(block.paragraphs):
            # Paragraph-level confidence as fallback when symbols lack scores
            para_conf = getattr(para, "confidence", 0.0) or 0.0

            for word in para.words:
                # ── Word text: join symbol texts ──────────────────────────────
                symbols = list(word.symbols)
                if not symbols:
                    continue
                text = "".join(s.text for s in symbols if s.text)
                if not text.strip():
                    continue

                # ── Word confidence: mean of symbol confidences ───────────────
                sym_confs = [
                    s.confidence for s in symbols
                    if hasattr(s, "confidence") and s.confidence is not None
                ]
                confidence = float(sum(sym_confs) / len(sym_confs)) if sym_confs else para_conf

                # ── Bounding box ──────────────────────────────────────────────
                if not word.bounding_box or not word.bounding_box.vertices:
                    log.debug("Word '%s' has no bounding box — skipping.", text)
                    continue

                vertices = word.bounding_box.vertices
                try:
                    bbox    = _bbox_from_vertices(vertices)
                    polygon = _polygon_from_vertices(vertices)
                except Exception as exc:
                    log.warning("Could not parse bbox for word '%s': %s", text, exc)
                    continue

                # ── Confidence filter ─────────────────────────────────────────
                if confidence < CLOUD_VISION_MIN_CONFIDENCE:
                    log.debug(
                        "Word '%s' confidence %.2f below threshold %.2f — skipped.",
                        text, confidence, CLOUD_VISION_MIN_CONFIDENCE,
                    )
                    continue

                words.append(WordEntry(
                    text=text,
                    confidence=round(confidence, 4),
                    bbox=bbox,
                    polygon=polygon,
                    block_id=block_id,
                    para_id=para_id,
                ))

    log.info(
        "Parsed label '%s': %d words, %d regions (img %d×%d).",
        label_id, len(words), len(regions), img_w, img_h,
    )

    return SemanticMap(
        label_id=label_id,
        image_width=img_w,
        image_height=img_h,
        words=words,
        regions=regions,
        raw_response=MessageToDict(response._pb),
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def extract(image_bytes: bytes, label_id: str) -> SemanticMap:
    """
    Send image_bytes to Cloud Vision DOCUMENT_TEXT_DETECTION and return a
    normalised SemanticMap.

    Called identically for:
      - Tier 1 (reference registration): result cached in Redis/Postgres.
      - Tier 2 (scan time): result compared against cached reference map.

    The Cloud Vision SDK call is synchronous (gRPC under the hood). We run
    it in the default thread executor so it does not block the asyncio event
    loop while waiting for the network response.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG/WEBP).
                     Must be < CLOUD_VISION_MAX_BYTES (20 MB API limit).
        label_id:    Identifier embedded in the returned SemanticMap.

    Returns:
        SemanticMap with words, layout regions, and raw Cloud Vision JSON.

    Raises:
        ValueError:   Image exceeds API size limit.
        RuntimeError: Cloud Vision returned an API-level error.
        google.auth.exceptions.DefaultCredentialsError: No credentials found.
    """
    if len(image_bytes) > CLOUD_VISION_MAX_BYTES:
        raise ValueError(
            f"Image too large: {len(image_bytes):,} bytes "
            f"(limit {CLOUD_VISION_MAX_BYTES:,})"
        )

    # ── Decode image dimensions (needed for SemanticMap metadata) ─────────────
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image bytes for label '{label_id}'.")
    img_h, img_w = img.shape[:2]

    # ── Build Cloud Vision request ────────────────────────────────────────────
    client  = _get_client()
    cv_image = vision.Image(content=image_bytes)
    feature  = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    request  = vision.AnnotateImageRequest(image=cv_image, features=[feature])

    log.info(
        "Calling Cloud Vision DOCUMENT_TEXT_DETECTION for label '%s' (%d×%d, %d bytes)…",
        label_id, img_w, img_h, len(image_bytes),
    )

    # ── Execute in thread executor (non-blocking) ─────────────────────────────
    loop     = asyncio.get_event_loop()
    response: AnnotateImageResponse = await loop.run_in_executor(
        None,
        client.annotate_image,
        request,
    )

    log.info("Cloud Vision response received for label '%s'.", label_id)

    return _parse_document_text_annotation(response, label_id, img_w, img_h)