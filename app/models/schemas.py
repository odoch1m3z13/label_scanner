"""
Pydantic schemas for all API request and response bodies.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class ScanVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


class ChangeType(str, Enum):
    TEXT = "text"
    LOGO = "logo"
    COLOR = "color"
    BARCODE = "barcode"
    ANOMALY = "anomaly"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


# ── Primitives ────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: int = Field(..., ge=0, description="Left pixel coordinate")
    y: int = Field(..., ge=0, description="Top pixel coordinate")
    w: int = Field(..., gt=0, description="Width in pixels")
    h: int = Field(..., gt=0, description="Height in pixels")

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_xywh(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


class Defect(BaseModel):
    change_type: ChangeType
    severity: Severity
    description: str
    ref_box: BoundingBox | None = None      # location on reference image
    scan_box: BoundingBox | None = None     # location on scanned image
    ref_value: str | None = None            # e.g. original text
    scan_value: str | None = None           # e.g. changed text
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    model_config = {"use_enum_values": True}


# ── Registration ──────────────────────────────────────────────────────────────

class RegisterResponse(BaseModel):
    label_id: str
    message: str
    stored_at: datetime
    preview_url: str


# ── Scan ─────────────────────────────────────────────────────────────────────

class StageResult(BaseModel):
    """Individual pipeline stage result."""
    stage: str
    duration_ms: float
    defects: list[Defect] = []
    metadata: dict[str, Any] = {}


class ScanRequest(BaseModel):
    label_id: str = Field(..., description="Reference label to compare against")


class ScanResponse(BaseModel):
    scan_id: str
    label_id: str
    verdict: ScanVerdict
    total_defects: int
    critical_defects: int
    stages: list[StageResult]
    all_defects: list[Defect]
    duration_ms: float
    scanned_at: datetime
    annotated_ref_url: str | None = None
    annotated_scan_url: str | None = None

    model_config = {"use_enum_values": True}


# ── Reference label metadata stored to disk ───────────────────────────────────

class ReferenceMetadata(BaseModel):
    label_id: str
    created_at: datetime
    image_path: str
    width: int
    height: int
    ocr_data: list[dict[str, Any]] = []
    logo_regions: list[dict[str, Any]] = []
    color_profile: list[dict[str, Any]] = []
    barcode_values: list[str] = []


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]