"""
Run this to diagnose storage connection issues:
    python check_env.py
"""
import asyncio, os, sys

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓  python-dotenv loaded .env")
except ImportError:
    print("✗  python-dotenv not installed — run: pip install python-dotenv")

redis_url    = os.environ.get("REDIS_URL")
postgres_dsn = os.environ.get("POSTGRES_DSN")
gac          = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

print()
print(f"REDIS_URL    = {redis_url!r}")
print(f"POSTGRES_DSN = {postgres_dsn!r}")
print(f"GOOGLE_APPLICATION_CREDENTIALS = {gac!r}")
print()

if not redis_url:
    print("✗  REDIS_URL is not set")
    sys.exit(1)
if not postgres_dsn:
    print("✗  POSTGRES_DSN is not set")
    sys.exit(1)

# ── Test Redis ────────────────────────────────────────────────────────────────
async def test_redis():
    import redis.asyncio as aioredis
    print("Testing Redis connection…")
    try:
        r = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await r.ping()
        await r.aclose()
        print("✓  Redis OK")
        return True
    except Exception as e:
        print(f"✗  Redis FAILED: {type(e).__name__}: {e}")
        return False

# ── Test Postgres ─────────────────────────────────────────────────────────────
async def test_postgres():
    import asyncpg
    print("Testing Postgres connection…")
    try:
        conn = await asyncpg.connect(postgres_dsn)
        await conn.fetchval("SELECT 1")
        await conn.close()
        print("✓  Postgres OK")
        return True
    except Exception as e:
        print(f"✗  Postgres FAILED: {type(e).__name__}: {e}")
        return False

async def main():
    r = await test_redis()
    p = await test_postgres()
    print()
    if r and p:
        print("✓  Both connections healthy — storage layer will initialise correctly.")
    else:
        print("✗  Fix the connection(s) above, then restart uvicorn.")

asyncio.run(main())