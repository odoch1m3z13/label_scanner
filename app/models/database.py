"""
Async SQLAlchemy ORM models and database initialisation.
"""

from __future__ import annotations
from sqlalchemy.pool import NullPool

import json
import uuid
from datetime import datetime

from sqlalchemy import String, Text, Float, Integer, DateTime, JSON
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

settings = get_settings()

# ── Engine & session factory ──────────────────────────────────────────────────

def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    # Supabase/PgBouncer: disable SQLAlchemy pooling entirely.
    # PgBouncer manages connections — holding them open causes timeout errors
    # on long-running requests (scans with model inference can take minutes).
    return {
        "poolclass": NullPool,
        "connect_args": {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex}__",
        },
    }

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    **_engine_kwargs(settings.database_url),
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM models ────────────────────────────────────────────────────────────────

class ReferenceLabel(Base):
    __tablename__ = "reference_labels"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    image_path: Mapped[str] = mapped_column(Text)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    # JSON blobs
    template_json: Mapped[str] = mapped_column(Text, default="[]")
    ocr_data: Mapped[str] = mapped_column(Text, default="[]")
    logo_regions: Mapped[str] = mapped_column(Text, default="[]")
    color_profile: Mapped[str] = mapped_column(Text, default="{}")
    barcode_values: Mapped[str] = mapped_column(Text, default="[]")

    def to_dict(self) -> dict:
        return {
            "label_id": self.id,
            "created_at": self.created_at.isoformat(),
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "template_json": self.template_json,
            "ocr_data": json.loads(self.ocr_data),
            "logo_regions": json.loads(self.logo_regions),
            "color_profile": json.loads(self.color_profile),
            "barcode_values": json.loads(self.barcode_values),
        }


class ScanResult(Base):
    __tablename__ = "scan_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label_id: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[str] = mapped_column(String(10))
    total_defects: Mapped[int] = mapped_column(Integer, default=0)
    critical_defects: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[float] = mapped_column(Float)
    scanned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    stages_json: Mapped[str] = mapped_column(Text, default="[]")
    defects_json: Mapped[str] = mapped_column(Text, default="[]")
    scan_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotated_ref_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotated_scan_path: Mapped[str | None] = mapped_column(Text, nullable=True)


# ── Initialisation ────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Dependency ────────────────────────────────────────────────────────────────

async def get_db() -> AsyncSession:  # type: ignore[misc]
    async with AsyncSessionLocal() as session:
        yield session