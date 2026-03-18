"""
Label Scanner – FastAPI application entry point.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import asyncio

from app.config import get_settings
from app.models.database import init_db

settings = get_settings()


# ── Logging 

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)
logger.add(
    "data/app.log",
    rotation="10 MB",
    retention="14 days",
    level="DEBUG",
    serialize=True,
)


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Label Scanner...")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir.parent / "scans").mkdir(parents=True, exist_ok=True)

    # Try DB immediately, then retry in background if it fails
    db_ready = await _try_init_db()
    if not db_ready:
        logger.warning("DB not ready — retrying in background every 30s")
        asyncio.create_task(_retry_db_loop())

    yield
    logger.info("Label Scanner shutting down")


async def _try_init_db() -> bool:
    """Attempt to initialise DB. Returns True on success, False on failure."""
    try:
        await init_db()
        logger.success("Database ready")
        return True
    except Exception as exc:
        logger.error(f"Database connection failed: {exc}")
        return False


async def _retry_db_loop():
    """Background task that retries DB connection every 30 seconds."""
    attempt = 0
    while True:
        await asyncio.sleep(30)
        attempt += 1
        logger.info(f"Retrying DB connection (attempt {attempt})...")
        success = await _try_init_db()
        if success:
            logger.success("Database connected successfully after retry")
            break
        if attempt >= 10:
            logger.error("DB connection failed after 10 attempts — giving up. Check DATABASE_URL in environment variables.")
            break# ── App ────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Industrial-grade label inspection API using a 6-stage CV pipeline.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers 

from app.api import health, register, scan  # noqa: E402

app.include_router(health.router)
app.include_router(register.router)
app.include_router(scan.router)

# ── Static files 

# /data MUST be mounted BEFORE / otherwise the catch-all "/" swallows image requests
data_dir = settings.base_dir / "data"
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

if settings.static_dir.exists():
    app.mount("/", StaticFiles(directory=str(settings.static_dir), html=True), name="static")