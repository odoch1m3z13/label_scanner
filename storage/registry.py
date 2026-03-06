"""
storage/registry.py — Label registry CRUD operations (Postgres).

Manages the labels table which is the system of record for:
  - Which label_ids are registered.
  - When they were first registered and last updated.
  - Scan history (counts, last scan timestamp).

Thin wrapper around asyncpg — all SQL is here, nowhere else.
"""
from __future__ import annotations

import logging
from datetime import datetime

log = logging.getLogger(__name__)


async def register_label(label_id: str, word_count: int, region_count: int) -> bool:
    """
    Insert or update a label record.
    Returns True if this was a new registration, False if an update.
    """
    # TODO: implement
    raise NotImplementedError


async def record_scan(
    label_id:        str,
    scan_id:         str,
    tamper_detected: bool,
    diff_counts:     dict,
) -> None:
    """
    Insert a scan result record into the scans table.
    Used for compliance audit trail.
    """
    # TODO: implement
    raise NotImplementedError


async def get_label_metadata(label_id: str) -> dict | None:
    """
    Return label registration metadata.
    Returns None if label_id is not registered.
    """
    # TODO: implement
    raise NotImplementedError


async def list_labels(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return paginated list of registered label_ids with metadata."""
    # TODO: implement
    raise NotImplementedError