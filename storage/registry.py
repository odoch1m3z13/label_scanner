"""
storage/registry.py — Label registry CRUD operations (Postgres).

Manages the labels and scans tables which are the system of record for:
  - Which label_ids are registered (labels table).
  - Registration metadata: word_count, region_count, scan statistics.
  - Every scan result for compliance audit trail (scans table).

Thin wrapper around asyncpg — all SQL lives here, nowhere else.

Pool sharing:
  registry.py shares the asyncpg pool created by storage.semantic_cache.init().
  Call registry.init(pg_pool) immediately after semantic_cache.init() returns.
  The pool is stored as a module-level singleton (_pg_pool) for use by all
  async functions in this module.

Schema:
  The labels table is first created by semantic_cache.init() with the base
  columns (label_id, semantic_map, registered_at, updated_at). registry.init()
  extends it with the metadata columns (word_count, region_count, scan_count,
  last_scanned_at) using ADD COLUMN IF NOT EXISTS, which is idempotent and
  safe to run on every startup against an existing production database.

  The scans table is created entirely by registry.init().

New vs update detection in register_label:
  Uses the PostgreSQL idiom `RETURNING (xmax = 0) AS is_new`. PostgreSQL sets
  xmax = 0 on a freshly inserted tuple and non-zero on an updated existing one.
  This allows a single round-trip UPSERT to determine whether the label was
  being registered for the first time.

Scan atomicity:
  record_scan() wraps its INSERT + UPDATE in a single asyncpg transaction so
  that a crash mid-way never leaves a scan record without its corresponding
  scan_count increment, or vice versa.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from config import PG_LABELS_TABLE, PG_SCANS_TABLE

log = logging.getLogger(__name__)

# Module-level pool singleton — set by init(), cleared by close().
_pg_pool: object | None = None


# =============================================================================
#  HELPERS
# =============================================================================

def _require_init() -> None:
    if _pg_pool is None:
        raise RuntimeError(
            "storage.registry not initialised — "
            "call registry.init(pg_pool) during FastAPI lifespan startup."
        )


def _row_to_dict(row) -> dict:
    """Convert an asyncpg Record to a plain Python dict."""
    return dict(row)


# =============================================================================
#  LIFECYCLE
# =============================================================================

async def init(pg_pool: object) -> None:
    """
    Receive the shared asyncpg pool from semantic_cache.init() and ensure
    the metadata columns and scans table exist.

    Called once during FastAPI lifespan startup, immediately after
    semantic_cache.init() has created the base labels table.

    Schema operations are idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS)
    so this is safe to run against an existing production database.

    Args:
        pg_pool: asyncpg connection pool (from semantic_cache._pg_pool).
    """
    global _pg_pool
    _pg_pool = pg_pool

    async with _pg_pool.acquire() as conn:
        # Extend labels table with registry-specific columns.
        # semantic_cache.init() creates the base table; we add metadata here.
        for col_ddl in (
            f"ALTER TABLE {PG_LABELS_TABLE} ADD COLUMN IF NOT EXISTS "
            f"word_count INT NOT NULL DEFAULT 0",
            f"ALTER TABLE {PG_LABELS_TABLE} ADD COLUMN IF NOT EXISTS "
            f"region_count INT NOT NULL DEFAULT 0",
            f"ALTER TABLE {PG_LABELS_TABLE} ADD COLUMN IF NOT EXISTS "
            f"scan_count INT NOT NULL DEFAULT 0",
            f"ALTER TABLE {PG_LABELS_TABLE} ADD COLUMN IF NOT EXISTS "
            f"last_scanned_at TIMESTAMPTZ",
            # Raw reference image bytes — needed by /compare for pixel-level diff.
            # Stored as BYTEA so it can be retrieved without re-uploading the ref.
            f"ALTER TABLE {PG_LABELS_TABLE} ADD COLUMN IF NOT EXISTS "
            f"ref_image_bytes BYTEA",
        ):
            await conn.execute(col_ddl)

        # Scans table — compliance audit trail
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {PG_SCANS_TABLE} (
                scan_id          TEXT PRIMARY KEY,
                label_id         TEXT NOT NULL,
                tamper_detected  BOOLEAN NOT NULL,
                diff_counts      JSONB NOT NULL,
                scanned_at       TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        # Index for per-label scan queries
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {PG_SCANS_TABLE}_label_id_idx
            ON {PG_SCANS_TABLE} (label_id, scanned_at DESC)
        """)

    log.info("Registry schema ready (%s, %s).", PG_LABELS_TABLE, PG_SCANS_TABLE)


async def close() -> None:
    """Release the pool reference. The pool itself is closed by semantic_cache.close()."""
    global _pg_pool
    _pg_pool = None
    log.info("Registry pool reference released.")


# =============================================================================
#  LABEL CRUD
# =============================================================================

async def register_label(
    label_id:     str,
    word_count:   int,
    region_count: int,
) -> bool:
    """
    Insert or update a label record with its word and region counts.

    Uses PostgreSQL's xmax=0 idiom to detect new vs update in a single
    round-trip UPSERT without a separate SELECT.

    Returns:
        True  — first registration (new label_id).
        False — update of an existing label (re-registration).
    """
    _require_init()

    async with _pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            INSERT INTO {PG_LABELS_TABLE}
                (label_id, semantic_map, word_count, region_count,
                 registered_at, updated_at)
            VALUES ($1, '{{}}'::jsonb, $2, $3, now(), now())
            ON CONFLICT (label_id) DO UPDATE
                SET word_count   = $2,
                    region_count = $3,
                    updated_at   = now()
            RETURNING (xmax = 0) AS is_new
            """,
            label_id,
            word_count,
            region_count,
        )

    is_new = bool(row["is_new"])
    log.info(
        "register_label: label_id=%s  is_new=%s  words=%d  regions=%d",
        label_id, is_new, word_count, region_count,
    )
    return is_new


# =============================================================================
#  SCAN AUDIT LOG
# =============================================================================

async def record_scan(
    label_id:        str,
    scan_id:         str,
    tamper_detected: bool,
    diff_counts:     dict,
) -> None:
    """
    Insert a scan result record and increment the label's scan counter.

    Executed inside a single asyncpg transaction so a failure mid-write
    never leaves the audit log in an inconsistent state.

    Args:
        label_id:        The registered label that was scanned.
        scan_id:         Unique identifier for this scan (UUID from caller).
        tamper_detected: True if any tamper box or diff flag was raised.
        diff_counts:     Dict mapping DiffType.value → count.
    """
    _require_init()

    diff_json = json.dumps(diff_counts)

    async with _pg_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                f"""
                INSERT INTO {PG_SCANS_TABLE}
                    (scan_id, label_id, tamper_detected, diff_counts, scanned_at)
                VALUES ($1, $2, $3, $4::jsonb, now())
                """,
                scan_id,
                label_id,
                tamper_detected,
                diff_json,
            )

            await conn.execute(
                f"""
                UPDATE {PG_LABELS_TABLE}
                SET scan_count      = scan_count + 1,
                    last_scanned_at = now()
                WHERE label_id = $1
                """,
                label_id,
            )

    log.info(
        "record_scan: label_id=%s  scan_id=%s  tamper=%s",
        label_id, scan_id, tamper_detected,
    )


# =============================================================================
#  METADATA QUERIES
# =============================================================================

async def get_label_metadata(label_id: str) -> dict | None:
    """
    Return registration metadata for a label.

    Does NOT return the semantic_map blob — only the lightweight metadata
    columns. Use semantic_cache.get() to retrieve the full SemanticMap.

    Returns:
        dict with keys: label_id, word_count, region_count, scan_count,
        last_scanned_at, registered_at, updated_at.
        None if label_id is not registered.
    """
    _require_init()

    async with _pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT label_id, word_count, region_count, scan_count,
                   last_scanned_at, registered_at, updated_at
            FROM {PG_LABELS_TABLE}
            WHERE label_id = $1
            """,
            label_id,
        )

    if row is None:
        return None

    meta = _row_to_dict(row)
    log.debug("get_label_metadata: label_id=%s found.", label_id)
    return meta


async def store_ref_image(label_id: str, image_bytes: bytes) -> None:
    """
    Persist the raw reference image bytes for a registered label.

    Called immediately after register_label() during POST /register.
    Stored as BYTEA so POST /compare can retrieve the pixels without requiring
    the caller to re-upload the reference image on every scan.

    Args:
        label_id:    Registered label identifier.
        image_bytes: Raw JPEG/PNG/WEBP bytes of the reference label image.
    """
    _require_init()

    async with _pg_pool.acquire() as conn:
        await conn.execute(
            f"""
            UPDATE {PG_LABELS_TABLE}
            SET ref_image_bytes = $2
            WHERE label_id = $1
            """,
            label_id,
            image_bytes,
        )

    log.debug("store_ref_image: label_id=%s  %d bytes.", label_id, len(image_bytes))


async def get_ref_image(label_id: str) -> bytes | None:
    """
    Retrieve the raw reference image bytes for a label.

    Returns None if the label is not registered or its image bytes have not
    been stored (e.g. a label registered before this column existed).

    Args:
        label_id: Registered label identifier.

    Returns:
        Raw image bytes, or None if not found.
    """
    _require_init()

    async with _pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT ref_image_bytes FROM {PG_LABELS_TABLE} WHERE label_id = $1",
            label_id,
        )

    if row is None or row["ref_image_bytes"] is None:
        log.debug("get_ref_image: label_id=%s — not found.", label_id)
        return None

    return bytes(row["ref_image_bytes"])


async def list_labels(limit: int = 100, offset: int = 0) -> list[dict]:
    """
    Return a paginated list of registered labels with metadata.

    Results are ordered by registered_at DESC (most recent first).
    Does NOT include the semantic_map blob.

    Args:
        limit:  Maximum number of rows to return (default 100, max enforced
                by caller — no hard cap here to allow admin tools to override).
        offset: Row offset for pagination.

    Returns:
        List of dicts, each with the same keys as get_label_metadata().
    """
    _require_init()

    # Clamp limit to a sane range — prevents runaway queries
    limit  = max(1, min(limit, 1000))
    offset = max(0, offset)

    async with _pg_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT label_id, word_count, region_count, scan_count,
                   last_scanned_at, registered_at, updated_at
            FROM {PG_LABELS_TABLE}
            ORDER BY registered_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )

    result = [_row_to_dict(r) for r in rows]
    log.debug("list_labels: limit=%d offset=%d → %d rows.", limit, offset, len(result))
    return result