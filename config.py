"""
config.py — Central configuration for Label Diff Scanner.
All tunable constants live here. Nothing in pipeline/ or workers/ should
hardcode numeric thresholds — import from this module instead.
"""
from __future__ import annotations

# ── Image processing ──────────────────────────────────────────────────────────
TAMPER_WORK_SIZE:     int = 1_000   # max px dimension for tamper working images
PREVIEW_MAX_PX:       int = 800     # max px dimension for frontend preview
PREVIEW_JPEG_QUALITY: int = 82

# ── Alignment  (pipeline/align.py) ───────────────────────────────────────────
ORB_N_FEATURES:              int   = 1_500
ORB_MIN_KEYPOINTS:           int   = 12
RANSAC_REPROJ_THRESHOLD:     float = 5.0
HOMOGRAPHY_INLIER_RATIO_HIGH: float = 0.85  # above this → tighten colour thresh

# ── Text diff  (pipeline/diff.py) ────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD:     float = 0.75
FUZZY_SHORT_TOKEN_MAX_LEN: int   = 4      # tokens ≤ 4 chars use stricter threshold
FUZZY_SHORT_THRESHOLD:     float = 0.80
FUZZY_EXACT_MAX_LEN:       int   = 2      # tokens ≤ 2 chars must match exactly
MIN_TOKEN_LEN:             int   = 3      # shorter tokens excluded from diff
MAX_CONSECUTIVE_FLAGS:     int   = 5      # run collapsed to unmatched_run above this
SPATIAL_TOLERANCE:         float = 0.12  # normalised 2D Euclidean for spatial pass
REFLOW_WINDOW:             int   = 8
REFLOW_CHAR_OVERLAP:       float = 0.75

# ── Colour change  (pipeline/colour.py) ──────────────────────────────────────
COLOR_DELTA_THRESHOLD:              float = 22.0
COLOR_DELTA_HIGH_INLIER_MULTIPLIER: float = 1.5
MIN_INK_PIXELS:                     int   = 25
MIN_BBOX_AREA_FOR_COLOR:            int   = 200
MIN_BBOX_H_FOR_COLOR:               int   = 15

# ── Visual tamper  (pipeline/tamper.py) ──────────────────────────────────────
MIN_TAMPER_AREA_FRAC:   float = 0.004
MAX_TAMPER_AREA_FRAC:   float = 0.35
TAMPER_COLOR_THRESH:    float = 18.0
GRAPHIC_CELL_FRAC:      float = 0.08
GRAPHIC_DIFF_THRESH:    float = 0.18
GRAPHIC_MIN_AREA_FRAC:  float = 0.003
TAMPER_MAX_ASPECT_RATIO: float = 8.0
TAMPER_ZONE_PAD_FRAC:   float = 0.20
SSIM_CELL_SIZE:         int   = 32
SSIM_TAMPER_THRESHOLD:  float = 0.75
DARK_PIXEL_L_THRESH:    float = 30.0     # L* below this → near-black (FIX-A)

# ── Barcode  (pipeline/barcode.py) ───────────────────────────────────────────
HOG_WIN_W:              int   = 128
HOG_WIN_H:              int   = 64
BARCODE_HOG_THRESH:     float = 0.72    # cosine sim below this → barcode changed
BARCODE_EDGE_DENSITY:   float = 0.12
BARCODE_MIN_WIDTH_FRAC:  float = 0.04
BARCODE_MIN_HEIGHT_FRAC: float = 0.02

# ── Layout  (pipeline/layout.py) ─────────────────────────────────────────────
LAYOUT_IOU_THRESHOLD: float = 0.30

# ── White border  (pipeline/border.py) ───────────────────────────────────────
WHITE_THRESHOLD:       int   = 240
MIN_BORDER_DELTA_PX:   int   = 20
MIN_BORDER_FRAC:       float = 0.02
BORDER_UNIFORMITY_FRAC: float = 0.85
BOUNDS_THUMB_PX:       int   = 1_000

# ── Cloud Vision  (workers/cloud_vision.py) ──────────────────────────────────
# DOCUMENT_TEXT_DETECTION returns full document hierarchy:
#   page → block → paragraph → word → symbol
# This is used for BOTH Tier 1 (reference registration) and Tier 2 (scan).
CLOUD_VISION_FEATURE:        str   = "DOCUMENT_TEXT_DETECTION"
CLOUD_VISION_MAX_BYTES:      int   = 20 * 1024 * 1024   # API hard limit
CLOUD_VISION_MIN_CONFIDENCE: float = 0.50

# ── Storage  (storage/) ──────────────────────────────────────────────────────
REDIS_SEMANTIC_KEY_PREFIX: str = "label:semantic:"
REDIS_SEMANTIC_TTL_S:      int = 7 * 24 * 3600   # 7 days
PG_LABELS_TABLE:           str = "labels"
PG_SCANS_TABLE:            str = "scans"