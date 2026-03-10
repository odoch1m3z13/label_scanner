"""
storage/semantic_cache.py — Semantic map persistence layer.

Two-tier storage strategy:
  - Redis: hot cache, sub-millisecond retrieval at scan time.
    Key:   label:semantic:{label_id}
    Value: JSON-serialised SemanticMap
    TTL:   REDIS_SEMANTIC_TTL_S (7 days, reset on every write)

  - Postgres: system of record, queried on Redis cache miss.
    Table: labels (label_id PK, semantic_map JSONB, registered_at, updated_at)

Cache miss flow:
  get(label_id)
    → Redis hit   → deserialise → return SemanticMap
    → Redis miss  → Postgres SELECT → re-warm Redis → return SemanticMap
    → Postgres miss → return None (label not registered)

Redis error resilience:
  A Redis error on get() falls through to Postgres transparently — the caller
  never sees the Redis failure. A Redis error on set() is logged but does not
  prevent the Postgres write from completing, since Postgres is the system of
  record. This means the worst case for a Redis outage is slower reads (Postgres
  latency instead of sub-ms), not data loss.

Serialisation:
  SemanticMap is serialised to/from JSON. Pydantic v1 (.json() / .parse_raw())
  and v2 (.model_dump_json() / .model_validate_json()) are both supported via
  the _to_json() / _from_json() helpers.

  asyncpg returns JSONB columns as Python dicts, not strings. The _from_pg_row()
  helper re-serialises the dict to a JSON string before passing to _from_json()
  so the same deserialisation path is used for both Redis and Postgres.

All functions are async (redis.asyncio + asyncpg).
Connection pools are module-level singletons set by init() during FastAPI
lifespan startup and torn down by close() during shutdown.
"""
from __future__ import annotations

import json
import logging

from config import (
    PG_LABELS_TABLE,
    REDIS_SEMANTIC_KEY_PREFIX,
    REDIS_SEMANTIC_TTL_S,
)
from models.schemas import SemanticMap

log = logging.getLogger(__name__)

# Module-level connection singletons — set by init(), cleared by close().
_redis:   object | None = None
_pg_pool: object | None = None


# =============================================================================
#  INTERNAL HELPERS
# =============================================================================

def _redis_key(label_id: str) -> str:
    """Build the Redis key for a label_id."""
    return f"{REDIS_SEMANTIC_KEY_PREFIX}{label_id}"


def _to_json(sm: SemanticMap) -> str:
    """
    Serialise a SemanticMap to a JSON string.
    Supports both Pydantic v1 (.json()) and v2 (.model_dump_json()).
    """
    try:
        return sm.model_dump_json()          # Pydantic v2
    except AttributeError:
        return sm.json()                     # Pydantic v1


def _from_json(raw: str) -> SemanticMap:
    """
    Deserialise a JSON string into a SemanticMap.
    Supports both Pydantic v1 (.parse_raw()) and v2 (.model_validate_json()).
    """
    try:
        return SemanticMap.model_validate_json(raw)   # Pydantic v2
    except AttributeError:
        return SemanticMap.parse_raw(raw)             # Pydantic v1


def _from_pg_row(row_value) -> SemanticMap:
    """
    Deserialise a SemanticMap from an asyncpg JSONB row value.

    asyncpg returns JSONB columns as Python dicts (already parsed), not as
    JSON strings. We re-serialise to a string so _from_json() can reconstruct
    the full Pydantic model with proper type coercion for nested objects.
    """
    if isinstance(row_value, str):
        return _from_json(row_value)
    return _from_json(json.dumps(row_value))


def _require_init() -> None:
    """Raise RuntimeError if init() has not been called."""
    if _redis is None or _pg_pool is None:
        raise RuntimeError(
            "storage.semantic_cache not initialised — "
            "call await semantic_cache.init(redis_url, postgres_dsn) "
            "during FastAPI lifespan startup before using get/set/delete/exists."
        )


# =============================================================================
#  LIFECYCLE
# =============================================================================

async def init(redis_url: str, postgres_dsn: str) -> None:
    """
    Initialise Redis and Postgres connection pools.
    Called once during FastAPI lifespan startup.

    Creates the labels table if it does not already exist.

    Args:
        redis_url:    Local:    "redis://localhost:6379/0"
                      Upstash:  "rediss://default:<pw>@<host>.upstash.io:6379"
                      (redis.asyncio handles TLS automatically from the rediss:// scheme)
        postgres_dsn: Local:    "postgresql://user:pass@localhost/labelscanner"
                      Supabase: "postgresql://postgres:<pw>@db.<ref>.supabase.co:5432/postgres"
                      (use port 5432 direct, NOT the Supabase pooler on port 6543)
    """
    global _redis, _pg_pool

    import redis.asyncio as aioredis
    import asyncpg

    _redis = await aioredis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )

    _pg_pool = await asyncpg.create_pool(postgres_dsn)

    # Ensure table exists — idempotent, safe to run on every startup
    async with _pg_pool.acquire() as conn:
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {PG_LABELS_TABLE} (
                label_id      TEXT PRIMARY KEY,
                semantic_map  JSONB NOT NULL,
                registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

    # Share pool with registry module (thin metadata layer, same DB)
    import storage.registry as _registry
    await _registry.init(_pg_pool)

    log.info("Storage layer ready (Redis + Postgres).")


async def close() -> None:
    """
    Close all connection pools.
    Called during FastAPI lifespan shutdown.
    """
    global _redis, _pg_pool

    # Release registry's pool reference first (pool itself closed below)
    try:
        import storage.registry as _registry
        await _registry.close()
    except Exception:
        log.exception("Error releasing registry pool reference.")

    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            log.exception("Error closing Redis connection.")
        _redis = None

    if _pg_pool is not None:
        try:
            await _pg_pool.close()
        except Exception:
            log.exception("Error closing Postgres pool.")
        _pg_pool = None

    log.info("Storage layer closed.")


# =============================================================================
#  CRUD
# =============================================================================

async def get(label_id: str) -> SemanticMap | None:
    """
    Retrieve a SemanticMap by label_id.

    Redis hit   → deserialise and return.
    Redis miss  → Postgres SELECT → re-warm Redis → return.
    Redis error → fall through to Postgres transparently (logged at WARNING).
    Not found   → return None.
    """
    _require_init()
    key = _redis_key(label_id)

    # ── Tier 1: Redis ──────────────────────────────────────────────────────
    redis_raw: str | None = None
    try:
        redis_raw = await _redis.get(key)
    except Exception:
        log.warning("Redis GET failed for key '%s' — falling through to Postgres.", key, exc_info=True)

    if redis_raw is not None:
        try:
            sm = _from_json(redis_raw)
            log.debug("Cache hit (Redis): label_id=%s", label_id)
            return sm
        except Exception:
            log.warning(
                "Failed to deserialise Redis value for label_id=%s — "
                "evicting and falling through to Postgres.",
                label_id, exc_info=True,
            )
            # Evict the corrupt entry so it doesn't keep blocking Postgres access
            try:
                await _redis.delete(key)
            except Exception:
                pass

    # ── Tier 2: Postgres ───────────────────────────────────────────────────
    log.debug("Cache miss (Redis): label_id=%s — querying Postgres.", label_id)
    try:
        async with _pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT semantic_map FROM {PG_LABELS_TABLE} WHERE label_id = $1",
                label_id,
            )
    except Exception:
        log.exception("Postgres SELECT failed for label_id=%s.", label_id)
        raise

    if row is None:
        log.debug("Cache miss (Postgres): label_id=%s not registered.", label_id)
        return None

    sm = _from_pg_row(row["semantic_map"])
    log.info("Cache hit (Postgres): label_id=%s — re-warming Redis.", label_id)

    # Re-warm Redis for the next scan
    try:
        await _redis.set(key, _to_json(sm), ex=REDIS_SEMANTIC_TTL_S)
    except Exception:
        log.warning("Redis SET (re-warm) failed for label_id=%s.", label_id, exc_info=True)

    return sm


async def set(semantic_map: SemanticMap) -> None:
    """
    Persist a SemanticMap to both Redis (hot) and Postgres (record).

    If label_id already exists in Postgres the row is updated (UPSERT).
    Redis TTL is reset on every write.

    Redis errors are logged but do NOT prevent the Postgres write —
    Postgres is the system of record.
    """
    _require_init()
    label_id = semantic_map.label_id
    key      = _redis_key(label_id)
    json_str = _to_json(semantic_map)

    # ── Postgres first (system of record) ─────────────────────────────────
    try:
        async with _pg_pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {PG_LABELS_TABLE}
                    (label_id, semantic_map, registered_at, updated_at)
                VALUES ($1, $2::jsonb, now(), now())
                ON CONFLICT (label_id) DO UPDATE
                    SET semantic_map = $2::jsonb,
                        updated_at   = now()
                """,
                label_id,
                json_str,
            )
    except Exception:
        log.exception("Postgres UPSERT failed for label_id=%s.", label_id)
        raise

    log.info("Postgres UPSERT: label_id=%s.", label_id)

    # ── Redis (hot cache, best-effort) ────────────────────────────────────
    try:
        await _redis.set(key, json_str, ex=REDIS_SEMANTIC_TTL_S)
        log.debug("Redis SET: key=%s  TTL=%ds", key, REDIS_SEMANTIC_TTL_S)
    except Exception:
        log.warning("Redis SET failed for label_id=%s — hot cache not updated.", label_id, exc_info=True)


async def delete(label_id: str) -> bool:
    """
    Remove a label from both Redis and Postgres.

    Returns True if the label existed in Postgres and was deleted.
    Redis deletion is best-effort (logged on error, does not affect return value).
    """
    _require_init()
    key = _redis_key(label_id)

    # ── Postgres (authoritative — determines return value) ─────────────────
    try:
        async with _pg_pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {PG_LABELS_TABLE} WHERE label_id = $1",
                label_id,
            )
    except Exception:
        log.exception("Postgres DELETE failed for label_id=%s.", label_id)
        raise

    # asyncpg returns "DELETE N" where N is the row count
    existed = result == "DELETE 1"
    log.info("Postgres DELETE: label_id=%s  existed=%s", label_id, existed)

    # ── Redis (best-effort) ────────────────────────────────────────────────
    try:
        await _redis.delete(key)
    except Exception:
        log.warning("Redis DELETE failed for label_id=%s.", label_id, exc_info=True)

    return existed


async def exists(label_id: str) -> bool:
    """
    Return True if label_id is registered (present in Redis OR Postgres).

    Checks Redis EXISTS first (O(1), cheapest path). On Redis miss or error,
    falls through to a Postgres COUNT query. Does NOT re-warm Redis — we do
    not have the full SemanticMap payload here, only the existence check.
    """
    _require_init()
    key = _redis_key(label_id)

    # ── Tier 1: Redis EXISTS ───────────────────────────────────────────────
    try:
        redis_exists = await _redis.exists(key)
        if redis_exists:
            log.debug("exists() Redis hit: label_id=%s", label_id)
            return True
    except Exception:
        log.warning("Redis EXISTS failed for label_id=%s — checking Postgres.", label_id, exc_info=True)

    # ── Tier 2: Postgres COUNT ────────────────────────────────────────────
    try:
        async with _pg_pool.acquire() as conn:
            count = await conn.fetchval(
                f"SELECT COUNT(1) FROM {PG_LABELS_TABLE} WHERE label_id = $1",
                label_id,
            )
    except Exception:
        log.exception("Postgres EXISTS check failed for label_id=%s.", label_id)
        raise

    found = bool(count)
    log.debug("exists() Postgres check: label_id=%s  found=%s", label_id, found)
    return found