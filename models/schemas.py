"""
models/schemas.py — Pydantic models for all API request/response payloads.

Typed models serve three purposes:
  1. Catch shape mismatches between pipeline stages at runtime.
  2. Auto-generate OpenAPI docs (/docs) with zero extra work.
  3. Make unit tests assertable against known field names/types.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
#  SHARED PRIMITIVES
# =============================================================================

class BoundingBox(BaseModel):
    """Axis-aligned bounding box in original image pixel coordinates."""
    x: int
    y: int
    w: int
    h: int


class Polygon(BaseModel):
    """Ordered list of (x, y) corner points — PaddleOCR / Cloud Vision format."""
    points: list[tuple[int, int]]


# =============================================================================
#  CLOUD VISION / SEMANTIC MAP
# =============================================================================

class WordEntry(BaseModel):
    """
    A single word as returned by Cloud Vision DOCUMENT_TEXT_DETECTION,
    normalised into our internal format.
    """
    text:        str
    confidence:  float
    bbox:        BoundingBox
    polygon:     Polygon
    block_id:    int   = Field(default=0, description="Cloud Vision block index")
    para_id:     int   = Field(default=0, description="Cloud Vision paragraph index")


class RegionType(str, Enum):
    text    = "text"
    figure  = "figure"
    table   = "table"
    barcode = "barcode"
    unknown = "unknown"


class LayoutRegion(BaseModel):
    """
    A spatial region identified by Cloud Vision or PP-Structure.
    Used for mask generation and targeted diff routing.
    """
    region_type: RegionType
    bbox:        BoundingBox
    label:       str | None = None   # e.g. "nutrition_panel", "logo", "barcode"


class SemanticMap(BaseModel):
    """
    Complete Tier 1 output for a single label image.
    Stored in Redis/Postgres at reference registration time.
    Retrieved at scan time to avoid re-running Cloud Vision on the reference.
    """
    label_id:     str
    image_width:  int
    image_height: int
    words:        list[WordEntry]
    regions:      list[LayoutRegion]
    raw_response: dict[str, Any] = Field(
        default_factory=dict,
        description="Full Cloud Vision JSON — preserved for reprocessing without API call",
    )


# =============================================================================
#  ALIGNMENT
# =============================================================================

class AlignmentResult(BaseModel):
    """Output of pipeline/align.py — homography matrix and quality metrics."""
    status:        str           # "ok" | "insufficient_keypoints" | "homography_failed"
    inlier_ratio:  float = 0.0
    inlier_count:  int   = 0
    total_matches: int   = 0
    # Homography matrices serialised as flat 9-element lists (row-major 3×3).
    # None when alignment failed.
    H:     list[float] | None = None
    H_inv: list[float] | None = None


# =============================================================================
#  DIFF ENTRIES
# =============================================================================

class DiffType(str, Enum):
    match         = "match"
    modified      = "modified"
    removed       = "removed"
    added         = "added"
    color_changed = "color_changed"
    unmatched_run = "unmatched_run"


class DiffEntry(BaseModel):
    """
    A single entry in the text diff output.
    The shape varies by type — optional fields are None when not applicable.
    """
    type: DiffType

    # Present for: match, modified, removed, color_changed
    ref:  WordEntry | None = None

    # Present for: match, modified, added, color_changed
    user: WordEntry | None = None

    # Present for: removed, added (single-word shorthand)
    word: WordEntry | None = None

    # Present for: modified
    similarity: float | None = None

    # Present for: color_changed
    color_delta: float | None = None

    # Present for: unmatched_run
    items: list[dict[str, Any]] | None = None
    count: int | None = None


# =============================================================================
#  TAMPER BOXES
# =============================================================================

class TamperSource(str, Enum):
    chroma      = "chroma"        # LAB pixel diff
    graphic     = "graphic"       # cell-grid SSIM diff
    barcode     = "barcode"       # HOG cosine diff
    layout      = "layout"        # PP-Structure / Cloud Vision figure mismatch
    white_border = "white_border" # padding delta


class TamperBox(BaseModel):
    """
    A flagged region in the reference or user image coordinate space.
    source identifies which detection subsystem raised the flag.
    """
    bbox:   BoundingBox
    source: TamperSource
    score:  float | None = None   # detector-specific confidence or delta value


# =============================================================================
#  API RESPONSES
# =============================================================================

class RegisterRequest(BaseModel):
    """Body for POST /register — registers a reference label."""
    label_id: str = Field(..., description="Unique identifier for this label SKU")


class RegisterResponse(BaseModel):
    """Response for POST /register."""
    label_id:    str
    word_count:  int
    region_count: int
    cached:      bool   # True if this label_id was already in cache (update)
    message:     str


class CompareResponse(BaseModel):
    """
    Full response for POST /compare.
    Returned to the frontend after a complete Tier 2 scan.
    """
    # ── Counts ────────────────────────────────────────────────────────────────
    removed_count:       int = 0
    added_count:         int = 0
    modified_count:      int = 0
    color_changed_count: int = 0
    unmatched_run_count: int = 0
    tamper_count_ref:    int = 0
    tamper_count_user:   int = 0
    white_border_count:  int = 0
    pp_figure_count:     int = 0

    # ── Diff payload ──────────────────────────────────────────────────────────
    diff: list[DiffEntry] = Field(default_factory=list)

    # ── Tamper boxes (ref coordinate space and user coordinate space) ─────────
    tamper_boxes_ref:  list[TamperBox] = Field(default_factory=list)
    tamper_boxes_user: list[TamperBox] = Field(default_factory=list)

    # ── Alignment metadata ────────────────────────────────────────────────────
    alignment: AlignmentResult | None = None

    # ── Image metadata ────────────────────────────────────────────────────────
    ref_size:        dict[str, int] | None = None   # {w, h}
    user_size:       dict[str, int] | None = None
    ref_word_count:  int = 0
    user_word_count: int = 0

    # ── Timing ───────────────────────────────────────────────────────────────
    phase1_time_s:  float = 0.0   # Cloud Vision + ORB parallel
    phase2_time_s:  float = 0.0   # diff + visual parallel
    total_time_s:   float = 0.0

    # ── Preview images (base64 JPEG for frontend) ─────────────────────────────
    ref_preview:  str | None = None
    user_preview: str | None = None


class HealthResponse(BaseModel):
    status:        str
    workers_ready: bool
    cache_ready:   bool
    version:       str = "5.0"