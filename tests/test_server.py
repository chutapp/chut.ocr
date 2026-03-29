"""OCR Server API tests."""

import io
import os

import pytest
from httpx import ASGITransport, AsyncClient

# Set test env before importing
os.environ["OCR_API_KEY"] = "test-key-for-ci"
os.environ["OPTII_INTERNAL_API_KEYS"] = "default:test-key-for-ci,test2:test-key-2"
os.environ["OPTII_DB_HOST"] = ""  # Disable DB for unit tests
os.environ["OPTII_DEMO_ENABLED"] = "true"

# Import the new app
from api.main import app

HEADERS = {"X-API-Key": "test-key-for-ci"}


def _transport(raise_errors: bool = True):
    return ASGITransport(app=app, raise_app_exceptions=raise_errors)


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_extract_no_auth():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.post("/ocr/extract")
        assert r.status_code in (401, 422)


@pytest.mark.asyncio
async def test_extract_wrong_key():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.post(
            "/ocr/extract",
            headers={"X-API-Key": "wrong"},
            files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_extract_bad_file_type():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.post(
            "/ocr/extract",
            headers=HEADERS,
            files={"file": ("test.exe", io.BytesIO(b"fake"), "application/octet-stream")},
        )
        assert r.status_code == 400
        assert "Unsupported file type" in r.json()["detail"]


@pytest.mark.asyncio
async def test_docs_disabled():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            r = await client.get(path)
            assert r.status_code in (401, 404), f"{path} should be disabled"


@pytest.mark.asyncio
async def test_demo_extract():
    """Demo endpoint should work without auth."""
    transport = _transport(raise_errors=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/demo/extract",
            files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
        )
        # Should not be 401 — demo requires no auth
        assert r.status_code != 401


@pytest.mark.asyncio
async def test_internal_key_works():
    """Internal API keys from env var should work."""
    transport = _transport(raise_errors=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/ocr/extract",
            headers={"X-API-Key": "test-key-2"},
            files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert r.status_code != 401, "Internal key test-key-2 should be accepted"


@pytest.mark.asyncio
async def test_auth_me_no_session():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.get("/auth/me")
        assert r.status_code == 401
