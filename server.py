"""
OCR Server — Secure API for document text extraction.

Agnostic: any project can call this API to extract text from images/PDFs.
Authentication: API key in X-API-Key header.
Model: PaddleOCR (multilingual: FR/NL/DE/EN).

Endpoints:
    POST /ocr/extract     — Extract text from uploaded image/PDF
    POST /ocr/invoice     — Extract structured invoice fields
    GET  /health          — Health check
"""

import hashlib
import hmac
import logging
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import PurePosixPath

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import API_KEYS, ALLOWED_ORIGINS, MAX_FILE_SIZE_MB, MAX_REQUESTS_PER_MINUTE, SUPPORTED_LANGUAGES

logger = logging.getLogger("ocr-server")
audit_logger = logging.getLogger("ocr-audit")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load OCR model on startup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger.info("OCR Server starting...")
    logger.info("Rate limit: %d/min", MAX_REQUESTS_PER_MINUTE)
    logger.info("Active API keys: %d", len(API_KEYS))
    try:
        _get_ocr()
        logger.info("PaddleOCR model loaded successfully")
    except Exception as e:
        logger.warning("PaddleOCR not available: %s", type(e).__name__)
        logger.warning("Server will start but OCR endpoints will fail")
    yield


app = FastAPI(
    title="OCR Server",
    description="Secure document text extraction API",
    version="1.2.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Rate limiting (per-IP, in-memory)
_request_log: dict[str, list[float]] = defaultdict(list)


def _get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request):
    client_ip = _get_client_ip(request)
    now = time.time()
    log = _request_log[client_ip]
    log[:] = [t for t in log if now - t < 60]
    if len(log) >= MAX_REQUESTS_PER_MINUTE:
        audit_logger.warning("rate_limit_exceeded ip=%s count=%d", client_ip, len(log))
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    log.append(now)


def _verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify API key against all active keys. Returns matched key ID for audit."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    for key_id, key_value in API_KEYS.items():
        if hmac.compare_digest(x_api_key, key_value):
            return key_id
    raise HTTPException(status_code=401, detail="Invalid API key")


def _validate_file_extension(filename: str | None) -> str:
    """Validate and return sanitized file suffix."""
    safe_name = PurePosixPath(filename).name if filename else ""
    suffix = PurePosixPath(safe_name).suffix.lower() if safe_name else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return suffix


# ─── PaddleOCR Initialization ───────────────────────────────────────────────

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            use_textline_orientation=True,
            lang="fr",
        )
    return _ocr_engine


# ─── Models ──────────────────────────────────────────────────────────────────


class OCRResult(BaseModel):
    success: bool
    text: str
    lines: list[dict]
    confidence: float
    processing_time_ms: int
    file_hash: str


class InvoiceFields(BaseModel):
    success: bool
    supplier_name: str = ""
    supplier_vat: str = ""
    invoice_number: str = ""
    invoice_date: str = ""
    total_excl_vat: str = ""
    total_incl_vat: str = ""
    vat_amount: str = ""
    currency: str = "EUR"
    raw_text: str = ""
    confidence: float = 0.0
    processing_time_ms: int = 0


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Health check — no auth required."""
    return {
        "status": "ok",
        "model": "PaddleOCR",
        "languages": SUPPORTED_LANGUAGES,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ocr/extract", response_model=OCRResult)
async def extract_text(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    """Extract raw text from an image or PDF. Returns lines with coordinates and confidence."""
    key_id = _verify_api_key(x_api_key)
    _check_rate_limit(request)
    suffix = _validate_file_extension(file.filename)
    request_id = uuid.uuid4().hex[:12]
    client_ip = _get_client_ip(request)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    file_hash = hashlib.sha256(content).hexdigest()[:16]
    start = time.time()

    audit_logger.info(
        "ocr_extract req=%s ip=%s key=%s file_hash=%s size_kb=%d type=%s",
        request_id, client_ip, key_id, file_hash, len(content) // 1024, suffix,
    )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete_on_close=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ocr = _get_ocr()
        results = ocr.ocr(tmp_path, cls=True)

        lines = []
        full_text_parts = []
        total_confidence = 0.0
        count = 0

        if results and results[0]:
            for line_result in results[0]:
                bbox = line_result[0]
                text = line_result[1][0]
                conf = line_result[1][1]

                lines.append({
                    "text": text,
                    "confidence": round(conf, 4),
                    "bbox": bbox,
                })
                full_text_parts.append(text)
                total_confidence += conf
                count += 1

        elapsed_ms = int((time.time() - start) * 1000)
        avg_confidence = total_confidence / count if count > 0 else 0.0

        audit_logger.info(
            "ocr_extract_done req=%s lines=%d confidence=%.4f ms=%d",
            request_id, count, avg_confidence, elapsed_ms,
        )

        return OCRResult(
            success=True,
            text="\n".join(full_text_parts),
            lines=lines,
            confidence=round(avg_confidence, 4),
            processing_time_ms=elapsed_ms,
            file_hash=file_hash,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR processing failed req=%s", request_id)
        raise HTTPException(status_code=500, detail="OCR processing failed")

    finally:
        import os
        os.unlink(tmp_path)


@app.post("/ocr/invoice", response_model=InvoiceFields)
async def extract_invoice(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    """Extract structured invoice fields from an image/PDF. Uses OCR + post-processing."""
    key_id = _verify_api_key(x_api_key)
    _check_rate_limit(request)
    suffix = _validate_file_extension(file.filename)
    request_id = uuid.uuid4().hex[:12]
    client_ip = _get_client_ip(request)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    start = time.time()

    audit_logger.info(
        "ocr_invoice req=%s ip=%s key=%s size_kb=%d type=%s",
        request_id, client_ip, key_id, len(content) // 1024, suffix,
    )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete_on_close=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ocr = _get_ocr()
        results = ocr.ocr(tmp_path, cls=True)

        raw_text = ""
        if results and results[0]:
            raw_text = "\n".join(line[1][0] for line in results[0])

        fields = _extract_invoice_fields(raw_text)
        elapsed_ms = int((time.time() - start) * 1000)

        audit_logger.info(
            "ocr_invoice_done req=%s ms=%d fields_found=%d",
            request_id, elapsed_ms, len(fields),
        )

        return InvoiceFields(
            success=True,
            raw_text=raw_text,
            processing_time_ms=elapsed_ms,
            **fields,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Invoice extraction failed req=%s", request_id)
        raise HTTPException(status_code=500, detail="Invoice extraction failed")

    finally:
        import os
        os.unlink(tmp_path)


def _extract_invoice_fields(text: str) -> dict:
    """Extract structured invoice fields from OCR text using regex patterns."""
    import re

    fields = {}

    # Supplier VAT (Belgian format)
    vat_match = re.search(r"(?:TVA|BTW|VAT)[:\s]*([A-Z]{2}\d{10})", text)
    if vat_match:
        fields["supplier_vat"] = vat_match.group(1)

    # Invoice number
    inv_match = re.search(r"(?:Facture|Invoice|Factuur)\s*(?:N[°o]?|#)?[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if inv_match:
        fields["invoice_number"] = inv_match.group(1)

    # Date
    date_match = re.search(r"(?:Date)[:\s]*(\d{2}[/.-]\d{2}[/.-]\d{4})", text, re.IGNORECASE)
    if date_match:
        fields["invoice_date"] = date_match.group(1)

    # Total amounts
    total_match = re.search(r"(?:Total\s*(?:TTC|incl|TVAC))[:\s]*(\d+[.,]\d{2})\s*(?:EUR)?", text, re.IGNORECASE)
    if total_match:
        fields["total_incl_vat"] = total_match.group(1).replace(",", ".")

    ht_match = re.search(r"(?:Total\s*(?:HT|excl|HTVA))[:\s]*(\d+[.,]\d{2})\s*(?:EUR)?", text, re.IGNORECASE)
    if ht_match:
        fields["total_excl_vat"] = ht_match.group(1).replace(",", ".")

    tva_match = re.search(r"(?:TVA|BTW|VAT)\s*\d+%[:\s]*(\d+[.,]\d{2})\s*(?:EUR)?", text, re.IGNORECASE)
    if tva_match:
        fields["vat_amount"] = tva_match.group(1).replace(",", ".")

    # Supplier name (first line that looks like a company name)
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) > 3 and not line.startswith(("FACTURE", "INVOICE", "Date", "TVA", "Total")):
            fields["supplier_name"] = line
            break

    return fields
