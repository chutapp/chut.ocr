"""API key management: create, list, revoke."""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from .auth import get_current_user
from .db import get_db
from .models import TIER_LIMITS, APIKeyInfo, CreateKeyRequest, CreateKeyResponse

logger = logging.getLogger("optii")

router = APIRouter(prefix="/keys", tags=["keys"])


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("", response_model=CreateKeyResponse)
async def create_key(req: CreateKeyRequest, request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    pool = get_db()
    tier = user["tier"]
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

    row = await pool.fetchrow(
        "SELECT COUNT(*) as cnt FROM api_keys WHERE user_id = $1 AND is_active = TRUE",
        user["id"],
    )
    if row["cnt"] >= limits["max_keys"]:
        raise HTTPException(
            status_code=403,
            detail=f"Key limit reached ({limits['max_keys']} for {tier} tier).",
        )

    raw_key = "optii_sk_" + secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)
    key_prefix = raw_key[:16]
    key_id = str(uuid.uuid4())
    now = _now_iso()

    await pool.execute(
        "INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, is_active, created_at) "
        "VALUES ($1, $2, $3, $4, $5, TRUE, $6)",
        key_id, user["id"], key_hash, key_prefix, req.name, now,
    )

    logger.info("API key created for user %s: %s...", user["email"], key_prefix)

    return CreateKeyResponse(
        key=raw_key,
        key_prefix=key_prefix,
        name=req.name,
        created_at=now,
    )


@router.get("", response_model=list[APIKeyInfo])
async def list_keys(request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    pool = get_db()
    rows = await pool.fetch(
        "SELECT id, key_prefix, name, is_active, created_at, last_used_at "
        "FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC",
        user["id"],
    )
    return [APIKeyInfo(**dict(r)) for r in rows]


@router.delete("/{key_id}")
async def revoke_key(key_id: str, request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    pool = get_db()
    result = await pool.execute(
        "UPDATE api_keys SET is_active = FALSE WHERE id = $1 AND user_id = $2 AND is_active = TRUE",
        key_id, user["id"],
    )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Key not found")

    logger.info("API key revoked: %s", key_id[:8])
    return {"ok": True}


async def lookup_api_key(token: str) -> dict | None:
    """Look up a database-backed API key. Returns key info or None."""
    pool = get_db()
    if pool is None:
        return None
    key_hash = _hash_key(token)
    row = await pool.fetchrow(
        "SELECT ak.id as key_id, ak.user_id, ak.name, u.tier "
        "FROM api_keys ak JOIN users u ON ak.user_id = u.id "
        "WHERE ak.key_hash = $1 AND ak.is_active = TRUE",
        key_hash,
    )
    if not row:
        return None

    # Update last_used_at (fire-and-forget)
    try:
        await pool.execute(
            "UPDATE api_keys SET last_used_at = $1 WHERE id = $2",
            _now_iso(), row["key_id"],
        )
    except Exception:
        pass

    return dict(row)
