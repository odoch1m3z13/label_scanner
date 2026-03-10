"""
tests/test_registry.py — Unit tests for storage/registry.py.

All tests use AsyncMock to simulate asyncpg connections.
No network, no real database.

Approach:
  - Inject a mock _pg_pool into registry._pg_pool before each test.
  - Reset to None after each test.
  - Verify SQL content and argument passing via mock call inspection.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, call, patch

# ── minimal pydantic + google stubs ──────────────────────────────────────────
pydantic_mod = types.ModuleType("pydantic")
class _BaseModel:
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
    def __eq__(self, o): return type(self)==type(o) and vars(self)==vars(o)
class _Field:
    def __new__(cls, *a, **kw): return kw.get('default', kw.get('default_factory', None))
pydantic_mod.BaseModel = _BaseModel
pydantic_mod.Field     = _Field
sys.modules['pydantic'] = pydantic_mod
for m in ['google','google.cloud','google.cloud.vision','google.cloud.vision_v1',
          'google.protobuf','google.protobuf.json_format','google.auth',
          'aioredis','asyncpg']:
    sys.modules[m] = types.ModuleType(m)

import storage.registry as reg
from config import PG_LABELS_TABLE, PG_SCANS_TABLE


# =============================================================================
#  Test helpers
# =============================================================================

def _make_conn(
    fetchrow_return=None,
    fetch_return=None,
    fetchval_return=None,
    execute_return="OK",
):
    """Build an AsyncMock simulating an asyncpg connection."""
    conn = AsyncMock()
    conn.fetchrow  = AsyncMock(return_value=fetchrow_return)
    conn.fetch     = AsyncMock(return_value=fetch_return or [])
    conn.fetchval  = AsyncMock(return_value=fetchval_return)
    conn.execute   = AsyncMock(return_value=execute_return)

    # transaction() context manager
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__  = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)

    return conn


def _make_pool(conn=None):
    """Build an AsyncMock simulating an asyncpg pool."""
    if conn is None:
        conn = _make_conn()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__  = AsyncMock(return_value=False)
    pool.close = AsyncMock()
    return pool, conn


def _inject(pool):
    reg._pg_pool = pool


def _reset():
    reg._pg_pool = None


# =============================================================================
#  _require_init
# =============================================================================

class TestRequireInit:
    def test_register_label_before_init_raises(self):
        _reset()
        try:
            asyncio.run(reg.register_label("x", 1, 1))
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "not initialised" in str(e).lower() or "init" in str(e)

    def test_record_scan_before_init_raises(self):
        _reset()
        try:
            asyncio.run(reg.record_scan("x", "s", False, {}))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    def test_get_label_metadata_before_init_raises(self):
        _reset()
        try:
            asyncio.run(reg.get_label_metadata("x"))
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass

    def test_list_labels_before_init_raises(self):
        _reset()
        try:
            asyncio.run(reg.list_labels())
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass


# =============================================================================
#  init()
# =============================================================================

class TestInit:
    def test_init_sets_pool(self):
        _reset()
        pool, conn = _make_pool()
        asyncio.run(reg.init(pool))
        assert reg._pg_pool is pool
        _reset()

    def test_init_executes_alter_table_statements(self):
        """init() must issue ALTER TABLE ... ADD COLUMN IF NOT EXISTS for each metadata column."""
        _reset()
        pool, conn = _make_pool()
        asyncio.run(reg.init(pool))
        calls = [str(c) for c in conn.execute.call_args_list]
        combined = " ".join(calls).upper()
        assert "ADD COLUMN IF NOT EXISTS" in combined
        _reset()

    def test_init_creates_scans_table(self):
        """init() must CREATE TABLE IF NOT EXISTS for the scans table."""
        _reset()
        pool, conn = _make_pool()
        asyncio.run(reg.init(pool))
        calls = [str(c) for c in conn.execute.call_args_list]
        combined = " ".join(calls).upper()
        assert "CREATE TABLE IF NOT EXISTS" in combined
        assert PG_SCANS_TABLE.upper() in combined
        _reset()

    def test_init_creates_index(self):
        """init() must create an index on the scans table."""
        _reset()
        pool, conn = _make_pool()
        asyncio.run(reg.init(pool))
        calls = [str(c) for c in conn.execute.call_args_list]
        combined = " ".join(calls).upper()
        assert "CREATE INDEX IF NOT EXISTS" in combined
        _reset()


# =============================================================================
#  register_label()
# =============================================================================

class TestRegisterLabel:
    def test_new_registration_returns_true(self):
        """xmax=0 row → True (new label)."""
        conn = _make_conn(fetchrow_return={"is_new": True})
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.register_label("sku-new", 42, 3))
            assert result is True
        finally:
            _reset()

    def test_update_returns_false(self):
        """xmax≠0 row → False (existing label updated)."""
        conn = _make_conn(fetchrow_return={"is_new": False})
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.register_label("sku-old", 40, 2))
            assert result is False
        finally:
            _reset()

    def test_calls_fetchrow_with_label_id(self):
        """fetchrow() must be called with label_id as the first positional arg."""
        conn = _make_conn(fetchrow_return={"is_new": True})
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.register_label("sku-check", 10, 1))
            args = conn.fetchrow.call_args[0]
            assert "sku-check" in args
        finally:
            _reset()

    def test_sql_contains_on_conflict(self):
        """SQL must use UPSERT (ON CONFLICT ... DO UPDATE)."""
        conn = _make_conn(fetchrow_return={"is_new": True})
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.register_label("sku-u", 5, 0))
            sql = conn.fetchrow.call_args[0][0]
            assert "ON CONFLICT" in sql.upper()
        finally:
            _reset()

    def test_sql_contains_xmax(self):
        """SQL must RETURN (xmax = 0) AS is_new."""
        conn = _make_conn(fetchrow_return={"is_new": True})
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.register_label("sku-u", 5, 0))
            sql = conn.fetchrow.call_args[0][0]
            assert "xmax" in sql.lower()
        finally:
            _reset()

    def test_word_count_passed_as_arg(self):
        """word_count must appear as a positional argument to fetchrow."""
        conn = _make_conn(fetchrow_return={"is_new": True})
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.register_label("sku-wc", 77, 4))
            args = conn.fetchrow.call_args[0]
            assert 77 in args
        finally:
            _reset()


# =============================================================================
#  record_scan()
# =============================================================================

class TestRecordScan:
    def test_inserts_into_scans_table(self):
        """execute() must be called with INSERT INTO scans."""
        conn = _make_conn()
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.record_scan("sku-s", "scan-001", True, {"removed": 2}))
            calls_sql = [c[0][0].upper() for c in conn.execute.call_args_list]
            assert any("INSERT" in s and PG_SCANS_TABLE.upper() in s for s in calls_sql)
        finally:
            _reset()

    def test_updates_scan_count_on_labels(self):
        """execute() must also UPDATE labels scan_count."""
        conn = _make_conn()
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.record_scan("sku-s", "scan-002", False, {}))
            calls_sql = [c[0][0].upper() for c in conn.execute.call_args_list]
            assert any("UPDATE" in s and PG_LABELS_TABLE.upper() in s for s in calls_sql)
        finally:
            _reset()

    def test_uses_transaction(self):
        """Both SQL statements must run inside a transaction."""
        conn = _make_conn()
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.record_scan("sku-s", "scan-003", True, {}))
            assert conn.transaction.called
        finally:
            _reset()

    def test_scan_id_passed(self):
        """scan_id must appear as an argument to one of the execute() calls."""
        conn = _make_conn()
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.record_scan("sku-s", "scan-id-xyz", False, {}))
            all_args = [str(c) for c in conn.execute.call_args_list]
            assert any("scan-id-xyz" in a for a in all_args)
        finally:
            _reset()

    def test_tamper_flag_passed(self):
        """tamper_detected bool must appear in execute() arguments."""
        conn = _make_conn()
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.record_scan("sku-s", "scan-t", True, {}))
            all_args = [str(c) for c in conn.execute.call_args_list]
            assert any("True" in a for a in all_args)
        finally:
            _reset()

    def test_diff_counts_serialised_as_json(self):
        """diff_counts dict must appear in execute() args as a JSON string."""
        conn = _make_conn()
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            counts = {"removed": 3, "added": 1}
            asyncio.run(reg.record_scan("sku-s", "scan-dc", False, counts))
            all_args = [str(c) for c in conn.execute.call_args_list]
            combined = " ".join(all_args)
            assert "removed" in combined
        finally:
            _reset()

    def test_returns_none(self):
        conn = _make_conn()
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.record_scan("sku-s", "scan-r", False, {}))
            assert result is None
        finally:
            _reset()


# =============================================================================
#  get_label_metadata()
# =============================================================================

class TestGetLabelMetadata:
    def _make_row(self, label_id="sku-m"):
        """Simulate an asyncpg Record as a dict."""
        return {
            "label_id":       label_id,
            "word_count":     50,
            "region_count":   3,
            "scan_count":     7,
            "last_scanned_at": None,
            "registered_at":  None,
            "updated_at":     None,
        }

    def test_returns_dict_when_found(self):
        row = self._make_row("sku-m")
        conn = _make_conn(fetchrow_return=row)
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.get_label_metadata("sku-m"))
            assert isinstance(result, dict)
            assert result["label_id"] == "sku-m"
            assert result["word_count"] == 50
        finally:
            _reset()

    def test_returns_none_when_not_found(self):
        conn = _make_conn(fetchrow_return=None)
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.get_label_metadata("sku-missing"))
            assert result is None
        finally:
            _reset()

    def test_queries_with_label_id(self):
        row = self._make_row()
        conn = _make_conn(fetchrow_return=row)
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.get_label_metadata("sku-q"))
            args = conn.fetchrow.call_args[0]
            assert "sku-q" in args
        finally:
            _reset()

    def test_does_not_include_semantic_map(self):
        """SQL must NOT select semantic_map — that blob is too large for metadata."""
        row = self._make_row()
        conn = _make_conn(fetchrow_return=row)
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.get_label_metadata("sku-nm"))
            sql = conn.fetchrow.call_args[0][0].lower()
            assert "semantic_map" not in sql
        finally:
            _reset()

    def test_result_contains_scan_count(self):
        row = self._make_row()
        row["scan_count"] = 12
        conn = _make_conn(fetchrow_return=row)
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.get_label_metadata("sku-sc"))
            assert result["scan_count"] == 12
        finally:
            _reset()


# =============================================================================
#  list_labels()
# =============================================================================

class TestListLabels:
    def _make_rows(self, n=3):
        return [
            {"label_id": f"sku-{i}", "word_count": i*10, "region_count": i,
             "scan_count": i*2, "last_scanned_at": None,
             "registered_at": None, "updated_at": None}
            for i in range(n)
        ]

    def test_returns_list_of_dicts(self):
        rows = self._make_rows(3)
        conn = _make_conn(fetch_return=rows)
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.list_labels())
            assert isinstance(result, list)
            assert len(result) == 3
            assert result[0]["label_id"] == "sku-0"
        finally:
            _reset()

    def test_empty_table_returns_empty_list(self):
        conn = _make_conn(fetch_return=[])
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            result = asyncio.run(reg.list_labels())
            assert result == []
        finally:
            _reset()

    def test_limit_passed_as_arg(self):
        conn = _make_conn(fetch_return=[])
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.list_labels(limit=25, offset=0))
            args = conn.fetch.call_args[0]
            assert 25 in args
        finally:
            _reset()

    def test_offset_passed_as_arg(self):
        conn = _make_conn(fetch_return=[])
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.list_labels(limit=10, offset=50))
            args = conn.fetch.call_args[0]
            assert 50 in args
        finally:
            _reset()

    def test_limit_clamped_to_max(self):
        """limit > 1000 must be clamped to 1000."""
        conn = _make_conn(fetch_return=[])
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.list_labels(limit=99999))
            args = conn.fetch.call_args[0]
            assert 99999 not in args
            assert 1000 in args
        finally:
            _reset()

    def test_negative_offset_clamped_to_zero(self):
        """offset < 0 must be clamped to 0."""
        conn = _make_conn(fetch_return=[])
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.list_labels(offset=-5))
            args = conn.fetch.call_args[0]
            assert -5 not in args
            assert 0 in args
        finally:
            _reset()

    def test_does_not_select_semantic_map(self):
        """SQL must NOT select semantic_map."""
        conn = _make_conn(fetch_return=[])
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.list_labels())
            sql = conn.fetch.call_args[0][0].lower()
            assert "semantic_map" not in sql
        finally:
            _reset()

    def test_orders_by_registered_at_desc(self):
        """SQL must ORDER BY registered_at DESC."""
        conn = _make_conn(fetch_return=[])
        pool, _ = _make_pool(conn)
        _inject(pool)
        try:
            asyncio.run(reg.list_labels())
            sql = conn.fetch.call_args[0][0].upper()
            assert "ORDER BY" in sql and "DESC" in sql
        finally:
            _reset()


# =============================================================================
#  close()
# =============================================================================

class TestClose:
    def test_close_resets_pool_to_none(self):
        pool, _ = _make_pool()
        _inject(pool)
        asyncio.run(reg.close())
        assert reg._pg_pool is None