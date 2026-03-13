"""
/api/register – Register a reference label.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import get_db
from app.models.schemas import RegisterResponse
from app.services import reference_store

settings = get_settings()
router = APIRouter(prefix="/api", tags=["Registration"])


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a reference label image",
)
async def register_label(
    file: UploadFile = File(..., description="Reference label image (PNG/JPG/BMP/TIFF)"),
    label_id: str | None = Form(None, description="Optional custom label ID"),
    db: AsyncSession = Depends(get_db),
) -> RegisterResponse:
    """
    Upload a reference label.  
    The system will run OCR, detect logo regions, build a colour profile,
    and decode any barcodes — all stored for comparison during scans.
    """
    # Validate content type
    allowed = {"image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type}",
        )

    # Validate size
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {settings.max_upload_mb} MB)",
        )

    try:
        meta = await reference_store.register_label(data, db, label_id=label_id)
    except Exception as exc:
        logger.exception(f"Label registration failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {exc}",
        )

    return RegisterResponse(
        label_id=meta.label_id,
        message="Reference label registered successfully",
        stored_at=meta.created_at,
        preview_url=f"/data/labels/{meta.label_id}/reference.png",
    )


@router.get("/labels", summary="List all registered reference labels")
async def list_labels(db: AsyncSession = Depends(get_db)) -> list[str]:
    return await reference_store.list_labels(db)


@router.delete("/labels/{label_id}", summary="Delete a reference label")
async def delete_label(
    label_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    deleted = await reference_store.delete_label(label_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Label not found")
    return {"deleted": label_id}