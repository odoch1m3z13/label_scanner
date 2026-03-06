"""
workers/cloud_vision.py — Async Google Cloud Vision client.

Responsibilities:
  - Send an image to Cloud Vision DOCUMENT_TEXT_DETECTION.
  - Parse the full document hierarchy (page → block → paragraph → word).
  - Return a normalised SemanticMap used identically by Tier 1 and Tier 2.

Design notes:
  - Pure async — no ProcessPoolExecutor needed (network I/O releases the GIL).
  - Stateless — no module-level globals except the client singleton.
  - Used for BOTH reference registration and scan-time extraction to guarantee
    identical tokenisation between the two (no tokenisation mismatch).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config import (
    CLOUD_VISION_FEATURE,
    CLOUD_VISION_MAX_BYTES,
    CLOUD_VISION_MIN_CONFIDENCE,
)
from models.schemas import BoundingBox, LayoutRegion, Polygon, RegionType, SemanticMap, WordEntry

log = logging.getLogger(__name__)

# Cloud Vision client singleton — initialised once on first call.
_cv_client: object | None = None


def _get_client() -> object:
    """
    Return (or lazily initialise) the Cloud Vision ImageAnnotatorClient.
    Reads credentials from GOOGLE_APPLICATION_CREDENTIALS env var or
    Application Default Credentials.
    """
    global _cv_client
    if _cv_client is not None:
        return _cv_client

    # lazy import so that module-level import failures don't happen if
    # google-cloud-vision isn't installed (tests/mock environments etc).
    try:
        from google.cloud import vision
    except ImportError as exc:  # pragma: no cover - environment missing
        raise ImportError(
            "google-cloud-vision library is required for Cloud Vision support"
        ) from exc

    # create a singleton client; credentials are picked up from
    # GOOGLE_APPLICATION_CREDENTIALS or ADC as per the library.
    _cv_client = vision.ImageAnnotatorClient()
    return _cv_client


def _bbox_from_vertices(vertices: list) -> BoundingBox:
    """
    Convert Cloud Vision vertex list to BoundingBox.
    Vertices are [top-left, top-right, bottom-right, bottom-left].
    """
    # vertices are list of objects with .x and .y attributes; they may
    # be missing or None.  Build an axis-aligned bounding box around the
    # supplied points.
    xs: list[int] = []
    ys: list[int] = []
    for v in vertices:
        # google.protobuf.FieldValue returns 0 when unset but just in case
        try:
            x = int(v.x) if v.x is not None else 0
        except Exception:  # pragma: no cover - defensive
            x = 0
        try:
            y = int(v.y) if v.y is not None else 0
        except Exception:  # pragma: no cover - defensive
            y = 0
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return BoundingBox(x=0, y=0, w=0, h=0)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return BoundingBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y)


def _polygon_from_vertices(vertices: list) -> Polygon:
    """Convert Cloud Vision vertex list to Polygon."""
    # Each vertex has x/y; some may be missing.  Preserve the
    # ordering from Cloud Vision (top-left, top-right, bottom-right,
    # bottom-left is usual but we don't rely on it).
    pts: list[tuple[int, int]] = []
    for v in vertices:
        try:
            x = int(v.x) if v.x is not None else 0
        except Exception:  # defensive
            x = 0
        try:
            y = int(v.y) if v.y is not None else 0
        except Exception:
            y = 0
        pts.append((x, y))
    return Polygon(points=pts)


def _parse_document_text_annotation(
    response:  object,
    label_id:  str,
    img_w:     int,
    img_h:     int,
) -> SemanticMap:
    """
    Parse a Cloud Vision DOCUMENT_TEXT_DETECTION response into a SemanticMap.

    Cloud Vision document hierarchy:
      response.full_text_annotation.pages[0]
        .blocks[]          ← block_id
          .paragraphs[]    ← para_id
            .words[]       ← WordEntry
              .symbols[]   ← individual characters (not used directly)

    Each word has:
      .bounding_box.vertices  → polygon
      .confidence             → float
      .symbols[].text joined  → word text
    """
    words: list[WordEntry] = []
    # extract pages (normally there is only one for label images)
    pages = []
    try:
        pages = response.full_text_annotation.pages
    except Exception:  # pragma: no cover - defensive
        pages = []

    block_id = 0
    for page in pages:
        para_id = 0
        for block in getattr(page, "blocks", []):
            for para in getattr(block, "paragraphs", []):
                for word in getattr(para, "words", []):
                    # assemble word text from individual symbols
                    text = "".join(getattr(s, "text", "") for s in getattr(word, "symbols", []))
                    confidence = float(getattr(word, "confidence", 0.0))
                    if confidence < CLOUD_VISION_MIN_CONFIDENCE:
                        continue
                    bbox = _bbox_from_vertices(getattr(word.bounding_box, "vertices", []))
                    polygon = _polygon_from_vertices(getattr(word.bounding_box, "vertices", []))
                    words.append(WordEntry(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        polygon=polygon,
                        block_id=block_id,
                        para_id=para_id,
                    ))
                para_id += 1
            block_id += 1

    # convert raw protobuf response to a plain dict for storage
    try:
        from google.protobuf.json_format import MessageToDict

        raw = MessageToDict(response._pb if hasattr(response, "_pb") else response)
    except Exception:  # pragma: no cover - if protobuf missing or error
        raw = {}

    return SemanticMap(
        label_id=label_id,
        image_width=img_w,
        image_height=img_h,
        words=words,
        regions=[],
        raw_response=raw,
    )


async def extract(image_bytes: bytes, label_id: str) -> SemanticMap:
    """
    Send image_bytes to Cloud Vision and return a normalised SemanticMap.

    Called identically for:
      - Tier 1 (reference registration): result is cached in Redis/Postgres.
      - Tier 2 (scan time): result is compared against cached reference map.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG/WEBP). Must be < CLOUD_VISION_MAX_BYTES.
        label_id:    Identifier stored in the returned SemanticMap.

    Returns:
        SemanticMap with words, regions, and the raw Cloud Vision response.
    """
    if len(image_bytes) > CLOUD_VISION_MAX_BYTES:
        raise ValueError(
            f"Image too large for Cloud Vision API: "
            f"{len(image_bytes):,} bytes > {CLOUD_VISION_MAX_BYTES:,} limit"
        )

    # perform API limit check above; now send request
    import asyncio
    try:
        from google.cloud import vision
    except ImportError:
        raise ImportError("google-cloud-vision library is required to call extract()")

    client = _get_client()
    image = vision.Image(content=image_bytes)
    # allow config to drive feature type in case we want to switch later
    feature = vision.Feature(type_=CLOUD_VISION_FEATURE)
    request = vision.AnnotateImageRequest(image=image, features=[feature])

    # Cloud Vision client is blocking; run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, client.annotate_image, request)

    # decode image bytes to calculate original dimensions
    try:
        import cv2  # local import to avoid heavy dependency at module import

        img_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image bytes for size extraction")
        h, w = img.shape[:2]
    except ImportError:  # cv2 not installed
        # fail gracefully if opencv missing; height/width unknown
        h, w = 0, 0

    return _parse_document_text_annotation(response, label_id, w, h)