"""OCR Server API tests."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

# Set test API key before importing server
os.environ["OCR_API_KEY"] = "test-key-for-ci"

from server import app

API_KEY = "test-key-for-ci"
HEADERS = {"X-API-Key": API_KEY}


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "languages" in data


@pytest.mark.asyncio
async def test_extract_no_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/ocr/extract")
        # Should fail with 401 or 422 (missing key or missing file)
        assert r.status_code in (401, 422)


@pytest.mark.asyncio
async def test_extract_wrong_key():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/ocr/extract", headers={"X-API-Key": "wrong"})
        assert r.status_code in (401, 422)  # 422 if file validation runs first


@pytest.mark.asyncio
async def test_invoice_wrong_key():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/ocr/invoice", headers={"X-API-Key": "wrong"})
        assert r.status_code in (401, 422)  # 422 if file validation runs first
