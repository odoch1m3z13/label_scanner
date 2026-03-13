"""
Centralized configuration using pydantic-settings.
All values can be overridden via environment variables or a .env file.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "LabelScanner"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Storage ──────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data" / "labels"
    static_dir: Path = base_dir / "static"

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/label_scanner.db"

    # ── Redis / Celery ───────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── Pipeline thresholds ──────────────────────────────────────────────
    # Alignment
    alignment_min_matches: int = 10          # minimum good feature matches
    alignment_ransac_threshold: float = 4.0  # pixels

    # OCR
    ocr_confidence_threshold: float = 0.6   # drop low-confidence boxes
    ocr_iou_threshold: float = 0.5          # match ref/scan text boxes

    # Logo / graphic similarity
    logo_similarity_threshold: float = 0.92  # cosine similarity (CLIP)

    # Color
    color_delta_e_threshold: float = 5.0    # CIEDE2000 ΔE threshold

    # Anomaly
    anomaly_score_threshold: float = 0.55   # 0-1 normalised score
    anomaly_min_area: int = 200             # px² — ignore tiny noise blobs

    # Decision engine
    critical_fields: list[str] = [          # any change here = FAIL
        "product_name",
        "weight",
        "expiry_date",
        "batch_number",
        "barcode",
    ]

    # ── Models ───────────────────────────────────────────────────────────
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # ── API ──────────────────────────────────────────────────────────────
    max_upload_mb: int = 20
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()