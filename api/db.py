"""PostgreSQL database layer for users, API keys, and usage tracking."""

from __future__ import annotations

import logging

import asyncpg

from .config import settings

logger = logging.getLogger("optii")

_pool: asyncpg.Pool | None = None

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         CITEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL DEFAULT '',
    tier          TEXT NOT NULL DEFAULT 'free',
    is_admin      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL REFERENCES users(id),
    key_hash      TEXT NOT NULL,
    key_prefix    TEXT NOT NULL,
    name          TEXT NOT NULL DEFAULT '',
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT
);

CREATE TABLE IF NOT EXISTS usage_log (
    id            BIGSERIAL PRIMARY KEY,
    api_key_id    TEXT NOT NULL REFERENCES api_keys(id),
    endpoint      TEXT NOT NULL,
    file_size_kb  INTEGER NOT NULL DEFAULT 0,
    processing_ms INTEGER NOT NULL DEFAULT 0,
    status_code   INTEGER NOT NULL DEFAULT 200,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_key_date ON usage_log(api_key_id, created_at);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
"""


async def init_db() -> None:
    """Initialize database connection pool and create tables."""
    global _pool

    if not settings.db_host:
        logger.warning("No database configured (OPTII_DB_HOST empty). DB features disabled.")
        return

    import ssl as _ssl

    ssl_ctx = _ssl.create_default_context() if settings.db_ssl else None
    _pool = await asyncpg.create_pool(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        ssl=ssl_ctx,
    )
    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA)
    logger.info(
        "Database pool initialized (%s@%s:%d/%s)",
        settings.db_user, settings.db_host, settings.db_port, settings.db_name,
    )


async def close_db() -> None:
    """Close database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_db() -> asyncpg.Pool | None:
    """Get the database connection pool. Returns None if not configured."""
    return _pool
