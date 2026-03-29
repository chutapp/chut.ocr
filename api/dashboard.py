"""Dashboard API: usage stats and account overview."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request

from .auth import get_current_user
from .db import get_db
from .models import TIER_LIMITS, DailyUsage, DashboardOverview, UsageStats

logger = logging.getLogger("optii")

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_PERIODS = {
    "24h": 1,
    "7d": 7,
    "30d": 30,
}


@router.get("/overview", response_model=DashboardOverview)
async def overview(request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    pool = get_db()

    key_row = await pool.fetchrow(
        "SELECT COUNT(*) as cnt FROM api_keys WHERE user_id = $1 AND is_active = TRUE",
        user["id"],
    )

    key_rows = await pool.fetch(
        "SELECT id FROM api_keys WHERE user_id = $1", user["id"],
    )
    key_ids = [row["id"] for row in key_rows]

    total_requests = 0
    total_file_size_kb = 0
    if key_ids:
        usage_row = await pool.fetchrow(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(file_size_kb), 0) as size_kb "
            "FROM usage_log WHERE api_key_id = ANY($1)",
            key_ids,
        )
        total_requests = usage_row["cnt"]
        total_file_size_kb = usage_row["size_kb"]

    return DashboardOverview(
        tier=user["tier"],
        tier_limits=TIER_LIMITS.get(user["tier"], TIER_LIMITS["free"]),
        key_count=key_row["cnt"],
        total_requests=total_requests,
        total_file_size_kb=total_file_size_kb,
    )


@router.get("/usage", response_model=UsageStats)
async def usage(request: Request, period: str = "7d"):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    days = _PERIODS.get(period, 7)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    pool = get_db()

    key_rows = await pool.fetch(
        "SELECT id FROM api_keys WHERE user_id = $1", user["id"],
    )
    key_ids = [row["id"] for row in key_rows]

    total_requests = 0
    total_file_size_kb = 0
    daily: list[DailyUsage] = []

    if key_ids:
        totals = await pool.fetchrow(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(file_size_kb), 0) as size_kb "
            "FROM usage_log WHERE api_key_id = ANY($1) AND created_at >= $2",
            key_ids, since,
        )
        total_requests = totals["cnt"]
        total_file_size_kb = totals["size_kb"]

        rows = await pool.fetch(
            "SELECT LEFT(created_at, 10) as date, "
            "COUNT(*) as requests, "
            "COALESCE(SUM(file_size_kb), 0) as file_size_kb, "
            "COALESCE(AVG(processing_ms), 0) as avg_ms "
            "FROM usage_log WHERE api_key_id = ANY($1) AND created_at >= $2 "
            "GROUP BY LEFT(created_at, 10) ORDER BY date",
            key_ids, since,
        )
        daily = [
            DailyUsage(
                date=row["date"],
                requests=row["requests"],
                file_size_kb=row["file_size_kb"],
                avg_ms=round(row["avg_ms"], 1),
            )
            for row in rows
        ]

    return UsageStats(
        period=period,
        total_requests=total_requests,
        total_file_size_kb=total_file_size_kb,
        daily=daily,
    )
