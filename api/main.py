"""OCR Server — API entrypoint with auth, keys, dashboard, and OCR."""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .auth import router as auth_router
from .config import settings
from .dashboard import router as dashboard_router
from .db import close_db, init_db
from .keys import lookup_api_key, router as keys_router
from .models import TIER_LIMITS
from .ocr import load_model, router as ocr_router


# ─── Structured JSON logging ─────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.hostname = os.environ.get("HOSTNAME", "unknown")

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.") + f"{record.msecs:03.0f}Z",
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
            "logger": record.name,
            "node": self.hostname,
        }
        for key in ("request_id", "method", "path", "status", "duration_ms",
                     "client_ip", "tier", "key_prefix", "user_id", "endpoint"):
            val = getattr(record, key, None)
            if val is not None:
                log[key] = val
        if record.exc_info and record.exc_info[2]:
            log["error"] = self.formatException(record.exc_info)
        return json.dumps(log, default=str)


def _setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.handlers = [handler]
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)


_setup_logging()
logger = logging.getLogger("optii")


# ─── Rate limiting ───────────────────────────────────────────────────────

_RATE_BUCKET_MAX_IPS = 10_000
_rate_buckets: dict[str, list[float]] = {}
_demo_rate_buckets: dict[str, list[float]] = {}
_key_rate_buckets: dict[str, list[float]] = {}


def _check_rate_limit(identifier: str, buckets: dict, max_rpm: int) -> bool:
    now = time.monotonic()
    cutoff = now - 60.0

    if max_rpm <= 0:
        return True

    bucket = buckets.get(identifier)
    if bucket is not None:
        buckets[identifier] = bucket = [t for t in bucket if t > cutoff]
    else:
        if len(buckets) >= _RATE_BUCKET_MAX_IPS:
            oldest = min(buckets, key=lambda k: buckets[k][-1] if buckets[k] else 0)
            del buckets[oldest]
        bucket = []
        buckets[identifier] = bucket

    if len(bucket) >= max_rpm:
        return False
    bucket.append(now)
    return True


# ─── App lifecycle ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading OCR model...")
    t0 = time.perf_counter()
    try:
        load_model()
        logger.info("OCR model loaded in %.1fs", time.perf_counter() - t0)
    except Exception as e:
        logger.warning("OCR model not available: %s", type(e).__name__)
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="Optii OCR API",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# Routers
app.include_router(auth_router)
app.include_router(keys_router)
app.include_router(dashboard_router)
app.include_router(ocr_router)

# CORS
if settings.cors_origins:
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "Cookie"],
    )


# ─── Auth + rate limiting middleware ─────────────────────────────────────

_PUBLIC_PATHS = {"/health"}
_PUBLIC_PREFIXES = ("/auth/", "/keys", "/dashboard")
_DEMO_PATHS = {"/demo/extract", "/demo/invoice"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"

    # Public paths: no auth needed
    if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return await call_next(request)

    # Demo paths: rate-limited, no auth
    if path in _DEMO_PATHS:
        if not settings.demo_enabled:
            return Response(status_code=401, content="Demo disabled")
        if not _check_rate_limit(client_ip, _demo_rate_buckets, settings.demo_rate_limit_rpm):
            return Response(status_code=429, content="Demo rate limit exceeded")
        request.state.demo = True
        request.state.api_key_id = None
        request.state.tier = "free"
        request.state.tier_limits = TIER_LIMITS["free"]
        return await call_next(request)

    # OCR paths: require auth
    # Check Authorization header first, then X-API-Key
    auth_header = request.headers.get("Authorization", "")
    api_key_header = request.headers.get("X-API-Key", "")
    token = ""

    if auth_header:
        token = auth_header.removeprefix("Bearer ").strip()
    elif api_key_header:
        token = api_key_header

    if not token:
        return Response(status_code=401, content="Unauthorized")

    # 1. Check internal keys (env var, no DB hit)
    internal_key_id = settings.is_internal_key(token)
    if internal_key_id:
        if not _check_rate_limit(client_ip, _rate_buckets, settings.rate_limit_rpm):
            return Response(status_code=429, content="Rate limit exceeded")
        request.state.demo = False
        request.state.api_key_id = None
        request.state.user_id = None
        request.state.tier = "enterprise"
        request.state.tier_limits = TIER_LIMITS["enterprise"]
        return await call_next(request)

    # 2. Check DB-backed API keys
    key_info = await lookup_api_key(token)
    if key_info:
        tier = key_info["tier"]
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

        if not _check_rate_limit(key_info["key_id"], _key_rate_buckets, limits["max_rpm"]):
            return Response(status_code=429, content="Rate limit exceeded")

        # Check monthly quota
        if limits["monthly_quota"] > 0:
            from .db import get_db
            from datetime import datetime, timezone
            pool = get_db()
            month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0).isoformat()
            row = await pool.fetchrow(
                "SELECT COUNT(*) as cnt FROM usage_log WHERE api_key_id = $1 AND created_at >= $2",
                key_info["key_id"], month_start,
            )
            if row["cnt"] >= limits["monthly_quota"]:
                return Response(status_code=429, content="Monthly quota exceeded")

        request.state.demo = False
        request.state.api_key_id = key_info["key_id"]
        request.state.user_id = key_info["user_id"]
        request.state.tier = tier
        request.state.tier_limits = limits
        return await call_next(request)

    return Response(status_code=401, content="Invalid API key")


# ─── Observability middleware ────────────────────────────────────────────

_SKIP_LOG_PATHS = {"/health"}


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.error("Unhandled exception", extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": 500,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else "unknown",
        }, exc_info=True)
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Request-ID"] = request_id

    path = request.url.path
    if path not in _SKIP_LOG_PATHS:
        status = response.status_code
        extra = {
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status": status,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else "unknown",
        }
        tier = getattr(request.state, "tier", None)
        if tier:
            extra["tier"] = tier
        key_id = getattr(request.state, "api_key_id", None)
        if key_id:
            extra["key_prefix"] = key_id[:8]

        if status >= 500:
            logger.error("request", extra=extra)
        elif status >= 400:
            logger.warning("request", extra=extra)
        else:
            logger.info("request", extra=extra)

    return response
