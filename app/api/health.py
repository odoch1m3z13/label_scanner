"""
/api/health – Liveness and readiness probes.
"""

from __future__ import annotations

import importlib

from fastapi import APIRouter
from loguru import logger

from app.config import get_settings
from app.models.schemas import HealthResponse

settings = get_settings()
router = APIRouter(prefix="/api", tags=["Health"])

_OPTIONAL_DEPS = {
    "paddleocr": "paddleocr",
    "tesseract": "pytesseract",
    "open_clip": "open_clip",
    "torch": "torch",
    "pyzbar": "pyzbar",
    "kornia": "kornia",
    "skimage": "skimage",
}


def _check_dep(module: str) -> str:
    try:
        importlib.import_module(module)
        return "ok"
    except ImportError:
        return "missing"


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    components = {name: _check_dep(mod) for name, mod in _OPTIONAL_DEPS.items()}
    components["opencv"] = _check_dep("cv2")
    components["numpy"] = _check_dep("numpy")

    # Check DB
    try:
        from app.models.database import engine
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        components["database"] = "ok"
    except Exception as exc:
        components["database"] = f"error: {exc}"

    all_ok = all(v == "ok" for k, v in components.items() if k in ("opencv", "numpy", "database"))
    status = "healthy" if all_ok else "degraded"

    return HealthResponse(
        status=status,
        version=settings.app_version,
        components=components,
    )