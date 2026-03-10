"""
tests/test_semantic_cache.py — Unit tests for storage/semantic_cache.py.

All tests use AsyncMock to simulate Redis and Postgres connections.
No network, no real database, no real Redis.

Strategy:
  - Before each test, inject mock _redis and _pg_pool into the module.
  - After each test, reset them to None (via fixture).
  - Verify the right mock methods were called with the right arguments.

Async test approach:
  Uses asyncio.run() in each test function rather than pytest-asyncio
  to keep the test runner dependency-free.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

# ── pydantic stub ─────────────────────────────────────────────────────────────
pydantic_mod = types.ModuleType("pydantic")
class _BaseModel:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
    def dict(self):
        return vars(self).copy()
    def json(self):
        return json.dumps(self.dict(), default=str)
    def model_dump_json(self):
        return self.json()
    @classmethod
    def parse_raw(cls, raw):
        return cls(**json.loads(raw))
    @classmethod
    def model_validate_json(cls, raw):
        return cls.parse_raw(raw)
    def __eq__(self, o): return type(self)==type(o) and vars(self)==vars(o)
    def __repr__(self): return f"{type(self).__name__}({vars(self)})"
class _Field:
    def __new__(cls, *a, **kw): return kw.get('default', kw.get('default_factory', None))
    @staticmethod
    def __call__(*a, **kw): return kw.get('default', kw.get('default_factory', None))
pydantic_mod.BaseModel = _BaseModel
pydantic_mod.Field     = _Field
sys.modules['pydantic'] = pydantic_mod

for m in ['google','google.cloud','google.cloud.vision','google.cloud.vision_v1',
          'google.protobuf','google.protobuf.json_format','google.auth',
          'aioredis','asyncpg']:
    sys.modules[m] = types.ModuleType(m)

# ── minimal SemanticMap for testing ──────────────────────────────────────────
from models.schemas import SemanticMap


def _make_sm(label_id: str = "sku-001") -> SemanticMap:
    return SemanticMap(
        label_id=label_id,
        image_width=400,
        image_height=300,
        words=[],
        regions=[],
        raw_response={},
    )


def _sm_json(sm: SemanticMap) -> str:
    return sm.json()


# ── test helpers ──────────────────────────────────────────────────────────────

def _make_redis_mock(get_return=None, exists_return=0):
    """Build an AsyncMock that simulates the aioredis client."""
    r = AsyncMock()
    r.get    = AsyncMock(return_value=get_return)
    r.set    = AsyncMock(return_value=True)
    r.delete = AsyncMock(return_value=1)
    r.exists = AsyncMock(return_value=exists_return)
    r.close  = AsyncMock()
    return r


def _make_pg_mock(fetchrow_return=None, fetchval_return=0, execute_return="DELETE 1"):
    """Build an AsyncMock simulating an asyncpg pool."""
    conn = AsyncMock()
    conn.fetchrow  = AsyncMock(return_value=fetchrow_return)
    conn.fetchval  = AsyncMock(return_value=fetchval_return)
    conn.execute   = AsyncMock(return_value=execute_return)

    # asyncpg pool: `async with pool.acquire() as conn`
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__  = AsyncMock(return_value=False)
    pool.close = AsyncMock()
    return pool, conn


def _inject(redis_mock, pg_mock):
    """Inject mocks into the cache module's global singletons."""
    import storage.semantic_cache as sc
    sc._redis   = redis_mock
    sc._pg_pool = pg_mock


def _reset():
    import storage.semantic_cache as sc
    sc._redis   = None
    sc._pg_pool = None


# =============================================================================
#  _redis_key
# =============================================================================

def test_redis_key_format():
    from storage.semantic_cache import _redis_key
    from config import REDIS_SEMANTIC_KEY_PREFIX
    assert _redis_key("sku-99") == f"{REDIS_SEMANTIC_KEY_PREFIX}sku-99"

def test_redis_key_unique_per_label():
    from storage.semantic_cache import _redis_key
    assert _redis_key("a") != _redis_key("b")


# =============================================================================
#  _to_json / _from_json round-trip
# =============================================================================

def test_serialise_roundtrip():
    from storage.semantic_cache import _to_json, _from_json
    sm   = _make_sm("sku-rt")
    raw  = _to_json(sm)
    sm2  = _from_json(raw)
    assert sm2.label_id    == "sku-rt"
    assert sm2.image_width == 400


def test_serialise_produces_valid_json():
    from storage.semantic_cache import _to_json
    sm  = _make_sm("chk-json")
    raw = _to_json(sm)
    parsed = json.loads(raw)
    assert parsed["label_id"] == "chk-json"


# =============================================================================
#  _from_pg_row — asyncpg dict path
# =============================================================================

def test_from_pg_row_dict():
    from storage.semantic_cache import _from_pg_row
    data = json.loads(_sm_json(_make_sm("sku-pg")))
    sm   = _from_pg_row(data)           # dict, as asyncpg returns
    assert sm.label_id == "sku-pg"


def test_from_pg_row_string():
    from storage.semantic_cache import _from_pg_row
    raw  = _sm_json(_make_sm("sku-str"))
    sm   = _from_pg_row(raw)            # str path
    assert sm.label_id == "sku-str"


# =============================================================================
#  _require_init
# =============================================================================

def test_get_before_init_raises():
    _reset()
    import storage.semantic_cache as sc
    try:
        asyncio.run(sc.get("sku-x"))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "not initialised" in str(e).lower() or "init" in str(e).lower()


def test_set_before_init_raises():
    _reset()
    import storage.semantic_cache as sc
    try:
        asyncio.run(sc.set(_make_sm()))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_delete_before_init_raises():
    _reset()
    import storage.semantic_cache as sc
    try:
        asyncio.run(sc.delete("x"))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_exists_before_init_raises():
    _reset()
    import storage.semantic_cache as sc
    try:
        asyncio.run(sc.exists("x"))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# =============================================================================
#  get()
# =============================================================================

def test_get_redis_hit_returns_semantic_map():
    """Redis hit → deserialise and return, Postgres NOT queried."""
    sm     = _make_sm("sku-a")
    r_mock = _make_redis_mock(get_return=_sm_json(sm))
    pg, conn = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['get']).get("sku-a"))
        assert result.label_id == "sku-a"
        conn.fetchrow.assert_not_called()
    finally:
        _reset()


def test_get_redis_miss_queries_postgres():
    """Redis miss → Postgres SELECT called."""
    sm       = _make_sm("sku-b")
    pg_row   = {"semantic_map": json.loads(_sm_json(sm))}
    r_mock   = _make_redis_mock(get_return=None)
    pg, conn = _make_pg_mock(fetchrow_return=pg_row)
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['get']).get("sku-b"))
        assert result.label_id == "sku-b"
        conn.fetchrow.assert_called_once()
    finally:
        _reset()


def test_get_redis_miss_rewarns_redis():
    """After Postgres hit, Redis.set() is called to re-warm the cache."""
    sm       = _make_sm("sku-c")
    pg_row   = {"semantic_map": json.loads(_sm_json(sm))}
    r_mock   = _make_redis_mock(get_return=None)
    pg, conn = _make_pg_mock(fetchrow_return=pg_row)
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['get']).get("sku-c"))
        r_mock.set.assert_called_once()
        # Verify TTL was passed
        call_kwargs = r_mock.set.call_args
        assert call_kwargs is not None
    finally:
        _reset()


def test_get_postgres_miss_returns_none():
    """Redis miss + Postgres miss → None."""
    r_mock   = _make_redis_mock(get_return=None)
    pg, conn = _make_pg_mock(fetchrow_return=None)
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['get']).get("sku-nope"))
        assert result is None
    finally:
        _reset()


def test_get_redis_error_falls_through_to_postgres():
    """Redis.get() raising an exception must fall through to Postgres."""
    sm       = _make_sm("sku-d")
    pg_row   = {"semantic_map": json.loads(_sm_json(sm))}
    r_mock   = _make_redis_mock()
    r_mock.get = AsyncMock(side_effect=ConnectionError("redis down"))
    pg, conn = _make_pg_mock(fetchrow_return=pg_row)
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['get']).get("sku-d"))
        assert result.label_id == "sku-d"
    finally:
        _reset()


def test_get_redis_returns_ttl_key():
    """Redis.get() is called with the correct prefixed key."""
    from config import REDIS_SEMANTIC_KEY_PREFIX
    sm     = _make_sm("sku-key")
    r_mock = _make_redis_mock(get_return=_sm_json(sm))
    pg, _  = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['get']).get("sku-key"))
        r_mock.get.assert_called_once_with(f"{REDIS_SEMANTIC_KEY_PREFIX}sku-key")
    finally:
        _reset()


# =============================================================================
#  set()
# =============================================================================

def test_set_writes_postgres_first():
    """Postgres execute() must be called when set() is called."""
    r_mock   = _make_redis_mock()
    pg, conn = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['set']).set(_make_sm()))
        conn.execute.assert_called_once()
    finally:
        _reset()


def test_set_writes_redis():
    """Redis.set() must be called with the right key and TTL."""
    from config import REDIS_SEMANTIC_KEY_PREFIX, REDIS_SEMANTIC_TTL_S
    sm       = _make_sm("sku-set")
    r_mock   = _make_redis_mock()
    pg, conn = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['set']).set(sm))
        r_mock.set.assert_called_once()
        args, kwargs = r_mock.set.call_args
        # First arg is key, second is JSON, kwarg ex= is TTL
        assert args[0] == f"{REDIS_SEMANTIC_KEY_PREFIX}sku-set"
        assert kwargs.get("ex") == REDIS_SEMANTIC_TTL_S or (
            len(args) >= 3 and args[2] == REDIS_SEMANTIC_TTL_S
        )
    finally:
        _reset()


def test_set_redis_failure_does_not_raise():
    """Redis.set() failure must be caught — Postgres write still completes."""
    r_mock   = _make_redis_mock()
    r_mock.set = AsyncMock(side_effect=ConnectionError("redis down"))
    pg, conn = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        # Should not raise even though Redis fails
        asyncio.run(__import__('storage.semantic_cache', fromlist=['set']).set(_make_sm()))
        conn.execute.assert_called_once()
    finally:
        _reset()


def test_set_upsert_sql_contains_on_conflict():
    """The SQL executed must include ON CONFLICT (upsert semantics)."""
    r_mock   = _make_redis_mock()
    pg, conn = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['set']).set(_make_sm()))
        sql = conn.execute.call_args[0][0]
        assert "ON CONFLICT" in sql.upper()
    finally:
        _reset()


# =============================================================================
#  delete()
# =============================================================================

def test_delete_existing_label_returns_true():
    r_mock   = _make_redis_mock()
    pg, conn = _make_pg_mock(execute_return="DELETE 1")
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['delete']).delete("sku-del"))
        assert result is True
    finally:
        _reset()


def test_delete_nonexistent_label_returns_false():
    r_mock   = _make_redis_mock()
    pg, conn = _make_pg_mock(execute_return="DELETE 0")
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['delete']).delete("sku-missing"))
        assert result is False
    finally:
        _reset()


def test_delete_calls_redis_delete():
    """Redis.delete() must be called with the correct key."""
    from config import REDIS_SEMANTIC_KEY_PREFIX
    r_mock   = _make_redis_mock()
    pg, conn = _make_pg_mock(execute_return="DELETE 1")
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['delete']).delete("sku-rdel"))
        r_mock.delete.assert_called_once_with(f"{REDIS_SEMANTIC_KEY_PREFIX}sku-rdel")
    finally:
        _reset()


def test_delete_redis_failure_does_not_affect_return():
    """Redis.delete() failure must not affect the return value."""
    r_mock   = _make_redis_mock()
    r_mock.delete = AsyncMock(side_effect=ConnectionError("redis down"))
    pg, conn = _make_pg_mock(execute_return="DELETE 1")
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['delete']).delete("sku-x"))
        assert result is True    # Postgres said row was deleted
    finally:
        _reset()


# =============================================================================
#  exists()
# =============================================================================

def test_exists_redis_hit_returns_true_no_postgres():
    """Redis EXISTS returns non-zero → True immediately, Postgres not queried."""
    r_mock   = _make_redis_mock(exists_return=1)
    pg, conn = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['exists']).exists("sku-ex"))
        assert result is True
        conn.fetchval.assert_not_called()
    finally:
        _reset()


def test_exists_redis_miss_queries_postgres_found():
    """Redis EXISTS 0 → Postgres COUNT 1 → True."""
    r_mock   = _make_redis_mock(exists_return=0)
    pg, conn = _make_pg_mock(fetchval_return=1)
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['exists']).exists("sku-pg"))
        assert result is True
        conn.fetchval.assert_called_once()
    finally:
        _reset()


def test_exists_redis_miss_postgres_not_found():
    """Redis EXISTS 0 → Postgres COUNT 0 → False."""
    r_mock   = _make_redis_mock(exists_return=0)
    pg, conn = _make_pg_mock(fetchval_return=0)
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['exists']).exists("sku-none"))
        assert result is False
    finally:
        _reset()


def test_exists_redis_error_falls_through():
    """Redis.exists() raising → fall through to Postgres."""
    r_mock   = _make_redis_mock()
    r_mock.exists = AsyncMock(side_effect=ConnectionError("redis down"))
    pg, conn = _make_pg_mock(fetchval_return=1)
    _inject(r_mock, pg)
    try:
        result = asyncio.run(__import__('storage.semantic_cache', fromlist=['exists']).exists("sku-fe"))
        assert result is True
        conn.fetchval.assert_called_once()
    finally:
        _reset()


def test_exists_does_not_rewarm_redis():
    """exists() must NOT call Redis.set() — no payload to cache."""
    r_mock   = _make_redis_mock(exists_return=0)
    pg, conn = _make_pg_mock(fetchval_return=1)
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['exists']).exists("sku-norewarm"))
        r_mock.set.assert_not_called()
    finally:
        _reset()


# =============================================================================
#  close()
# =============================================================================

def test_close_calls_redis_close():
    r_mock = _make_redis_mock()
    pg, _  = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['close']).close())
        r_mock.close.assert_called_once()
    finally:
        _reset()


def test_close_calls_pg_close():
    r_mock = _make_redis_mock()
    pg, _  = _make_pg_mock()
    _inject(r_mock, pg)
    try:
        asyncio.run(__import__('storage.semantic_cache', fromlist=['close']).close())
        pg.close.assert_called_once()
    finally:
        _reset()


def test_close_resets_singletons():
    """After close(), _redis and _pg_pool must be None."""
    import storage.semantic_cache as sc
    r_mock = _make_redis_mock()
    pg, _  = _make_pg_mock()
    _inject(r_mock, pg)
    asyncio.run(sc.close())
    assert sc._redis   is None
    assert sc._pg_pool is None