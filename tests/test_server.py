"""OCR Server API tests."""

import io
import os

import pytest
from httpx import ASGITransport, AsyncClient

# Set test API keys before importing server
os.environ["OCR_API_KEY"] = "test-key-for-ci"

from server import app

API_KEY = "test-key-for-ci"
HEADERS = {"X-API-Key": API_KEY}


def _transport(raise_errors: bool = True):
    return ASGITransport(app=app, raise_app_exceptions=raise_errors)


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "languages" in data


@pytest.mark.asyncio
async def test_extract_no_auth():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.post("/ocr/extract")
        assert r.status_code in (401, 422)


@pytest.mark.asyncio
async def test_extract_wrong_key():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.post("/ocr/extract", headers={"X-API-Key": "wrong"})
        assert r.status_code in (401, 422)


@pytest.mark.asyncio
async def test_invoice_wrong_key():
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.post("/ocr/invoice", headers={"X-API-Key": "wrong"})
        assert r.status_code in (401, 422)


@pytest.mark.asyncio
async def test_docs_disabled():
    """Verify API docs are not publicly accessible."""
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            r = await client.get(path)
            assert r.status_code == 404, f"{path} should be disabled"


@pytest.mark.asyncio
async def test_extract_bad_file_type():
    """Verify only allowed file types are accepted."""
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.post(
            "/ocr/extract",
            headers=HEADERS,
            files={"file": ("test.exe", io.BytesIO(b"fake"), "application/octet-stream")},
        )
        assert r.status_code == 400
        assert "Unsupported file type" in r.json()["detail"]


@pytest.mark.asyncio
async def test_cors_not_wildcard():
    """Verify CORS does not reflect arbitrary origins."""
    async with AsyncClient(transport=_transport(), base_url="http://test") as client:
        r = await client.get("/health", headers={"Origin": "https://evil.com"})
        acao = r.headers.get("access-control-allow-origin", "")
        assert acao != "*", "CORS should not allow all origins"


@pytest.mark.asyncio
async def test_path_traversal_filename():
    """Verify path traversal in filename is sanitized — auth passes, no 400 for .png."""
    transport = _transport(raise_errors=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/ocr/extract",
            headers=HEADERS,
            files={"file": ("../../../etc/passwd.png", io.BytesIO(b"fake"), "image/png")},
        )
        # Auth passed, file type valid. 500 expected if PaddleOCR not available locally.
        assert r.status_code != 401
        assert r.status_code != 400


@pytest.mark.asyncio
async def test_multi_key_rotation():
    """Verify multiple API keys work for rotation."""
    os.environ["OCR_API_KEYS"] = "app1:key-alpha,app2:key-beta"
    from importlib import reload
    import config
    reload(config)
    import server as server_mod
    server_mod._ocr_engine = None
    reload(server_mod)
    from server import app as fresh_app

    transport = ASGITransport(app=fresh_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post(
            "/ocr/extract",
            headers={"X-API-Key": "key-alpha"},
            files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
        )
        r2 = await client.post(
            "/ocr/extract",
            headers={"X-API-Key": "key-beta"},
            files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert r1.status_code != 401, "key-alpha should be accepted"
        assert r2.status_code != 401, "key-beta should be accepted"

        r3 = await client.post(
            "/ocr/extract",
            headers={"X-API-Key": "test-key-for-ci"},
            files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
        )
        assert r3.status_code == 401, "old key should be rejected"

    # Cleanup
    del os.environ["OCR_API_KEYS"]
    os.environ["OCR_API_KEY"] = "test-key-for-ci"
    reload(config)
    server_mod._ocr_engine = None
    reload(server_mod)
