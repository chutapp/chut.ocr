"""Authentication via Chut Auth (OAuth2 + JWKS)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import jwt
from jwt import PyJWKClient
from fastapi import APIRouter, HTTPException, Request, Response

from .config import settings
from .db import get_db
from .models import AuthResponse, UserProfile

logger = logging.getLogger("optii")

router = APIRouter(prefix="/auth", tags=["auth"])

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(
            f"{settings.auth_url}/.well-known/jwks.json",
            cache_keys=True,
        )
    return _jwks_client


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _ensure_user(pool, user_id: str, email: str) -> None:
    """Create or migrate user record."""
    existing = await pool.fetchrow("SELECT id, tier FROM users WHERE email = $1", email)
    if existing and existing["id"] != user_id:
        old_id = existing["id"]
        old_tier = existing["tier"]
        await pool.execute(
            "INSERT INTO users (id, email, password_hash, tier, created_at) "
            "VALUES ($1, $2, '', $3, $4) ON CONFLICT (id) DO NOTHING",
            user_id, email + "_migrating", old_tier, _now_iso(),
        )
        await pool.execute("UPDATE api_keys SET user_id = $1 WHERE user_id = $2", user_id, old_id)
        await pool.execute("DELETE FROM users WHERE id = $1", old_id)
        await pool.execute("UPDATE users SET email = $1 WHERE id = $2", email, user_id)
    elif not existing:
        await pool.execute(
            "INSERT INTO users (id, email, password_hash, tier, created_at) "
            "VALUES ($1, $2, '', 'free', $3) ON CONFLICT (id) DO NOTHING",
            user_id, email, _now_iso(),
        )


def _verify_access_token(token: str) -> dict:
    """Verify a Chut Auth access token via JWKS. Returns claims."""
    client = _get_jwks_client()
    signing_key = client.get_signing_key_from_jwt(token)
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=settings.auth_url,
        audience=settings.auth_client_id,
    )
    if payload.get("type") != "access":
        raise ValueError("Not an access token")
    return payload


def _set_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    response.set_cookie(
        key="optii_access",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=3600,
    )
    response.set_cookie(
        key="optii_refresh",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
        max_age=30 * 24 * 3600,
    )


def decode_session(request: Request) -> dict | None:
    """Decode access token from cookie or Authorization header. Returns claims or None."""
    token = request.cookies.get("optii_access")
    if not token:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
    if not token:
        return None
    try:
        return _verify_access_token(token)
    except Exception:
        return None


async def get_current_user(request: Request) -> dict | None:
    """Get current user from access token cookie. Returns user dict or None."""
    claims = decode_session(request)
    if not claims:
        claims = await _try_refresh(request)
        if not claims:
            return None

    user_id = claims["sub"]
    email = claims.get("email", "")

    pool = get_db()
    row = await pool.fetchrow(
        "SELECT id, email, tier, is_admin, created_at FROM users WHERE id = $1",
        user_id,
    )
    if not row:
        await _ensure_user(pool, user_id, email)
        return {"id": user_id, "email": email, "tier": "free", "is_admin": False, "created_at": _now_iso()}

    return dict(row)


async def _try_refresh(request: Request) -> dict | None:
    """Try to refresh the access token using the refresh cookie."""
    refresh_token = request.cookies.get("optii_refresh")
    if not refresh_token:
        return None
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.auth_url}/refresh",
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": settings.auth_client_id,
                    "client_secret": settings.auth_client_secret,
                },
                timeout=10,
            )
        if resp.status_code != 200:
            return None
        tokens = resp.json()
        request.state._new_tokens = tokens
        return _verify_access_token(tokens["access_token"])
    except Exception:
        return None


@router.post("/register-user")
async def register_proxy(request: Request):
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.auth_url}/register",
            json={"email": body.get("email"), "password": body.get("password"), "name": ""},
            timeout=10,
        )
    if resp.status_code == 409:
        raise HTTPException(status_code=409, detail="Email already registered")
    if resp.status_code not in (200, 201):
        detail = resp.json().get("detail", "Registration failed")
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


@router.post("/authorize")
async def authorize_proxy(request: Request):
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.auth_url}/authorize",
            json={
                "email": body.get("email"),
                "password": body.get("password"),
                "client_id": settings.auth_client_id,
                "redirect_uri": body.get("redirect_uri", ""),
                "state": body.get("state", ""),
            },
            timeout=10,
        )
    if resp.status_code != 200:
        detail = resp.json().get("detail", "Login failed")
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


@router.post("/callback")
async def callback(request: Request, response: Response):
    body = await request.json()
    code = body.get("code")
    state = body.get("state")
    redirect_uri = body.get("redirect_uri")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")

    saved_state = request.cookies.get("oauth_state")
    if not saved_state or saved_state != state:
        raise HTTPException(status_code=403, detail="Invalid state")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.auth_url}/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": settings.auth_client_id,
                "client_secret": settings.auth_client_secret,
            },
            timeout=10,
        )

    if resp.status_code != 200:
        detail = resp.json().get("detail", "Token exchange failed")
        raise HTTPException(status_code=401, detail=detail)

    tokens = resp.json()
    claims = _verify_access_token(tokens["access_token"])

    pool = get_db()
    await _ensure_user(pool, claims["sub"], claims.get("email", ""))

    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "user_id": claims["sub"],
        "email": claims.get("email", ""),
    }


@router.post("/otp/request")
async def otp_request(request: Request):
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.auth_url}/otp/request",
            json={
                "email": body.get("email"),
                "client_id": settings.auth_client_id,
                "redirect_uri": body.get("redirect_uri", ""),
            },
            timeout=10,
        )
    if resp.status_code != 200:
        detail = resp.json().get("detail", "Failed to send code")
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


@router.post("/otp/verify")
async def otp_verify(request: Request, response: Response):
    body = await request.json()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.auth_url}/otp/verify",
            json={
                "email": body.get("email"),
                "code": body.get("code"),
                "client_id": settings.auth_client_id,
                "redirect_uri": body.get("redirect_uri", ""),
            },
            timeout=10,
        )
    if resp.status_code != 200:
        detail = resp.json().get("detail", "Invalid code")
        raise HTTPException(status_code=resp.status_code, detail=detail)

    tokens = resp.json()
    claims = _verify_access_token(tokens["access_token"])

    pool = get_db()
    await _ensure_user(pool, claims["sub"], claims.get("email", ""))

    _set_cookies(response, tokens["access_token"], tokens["refresh_token"])
    return {
        "ok": True,
        "user_id": claims["sub"],
        "email": claims.get("email", ""),
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
    }


@router.post("/otp-callback")
async def otp_callback(request: Request, response: Response):
    body = await request.json()
    access_token = body.get("access_token")
    refresh_token = body.get("refresh_token")

    if not access_token or not refresh_token:
        raise HTTPException(status_code=400, detail="Missing tokens")

    claims = _verify_access_token(access_token)

    pool = get_db()
    await _ensure_user(pool, claims["sub"], claims.get("email", ""))

    _set_cookies(response, access_token, refresh_token)
    return {"ok": True, "user_id": claims["sub"], "email": claims.get("email", "")}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("optii_access", path="/")
    response.delete_cookie("optii_refresh", path="/")
    return {"ok": True}


@router.get("/me", response_model=UserProfile)
async def me(request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    pool = get_db()
    row = await pool.fetchrow(
        "SELECT COUNT(*) as cnt FROM api_keys WHERE user_id = $1 AND is_active = TRUE",
        user["id"],
    )

    return UserProfile(
        id=user["id"],
        email=user["email"],
        tier=user["tier"],
        created_at=user["created_at"],
        key_count=row["cnt"],
    )
