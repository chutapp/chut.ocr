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
import io
import tempfile
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import API_KEY, MAX_FILE_SIZE_MB, MAX_REQUESTS_PER_MINUTE, SUPPORTED_LANGUAGES

app = FastAPI(
    title="OCR Server",
    description="Secure document text extraction API",
    version="1.0.0",
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domains in production
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Rate limiting (simple in-memory)
_request_log: list[float] = []


def _check_rate_limit():
    now = time.time()
    _request_log[:] = [t for t in _request_log if now - t < 60]
    if len(_request_log) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    _request_log.append(now)


def _verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ─── PaddleOCR Initialization ───────────────────────────────────────────────

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="fr",  # Primary language; handles multilingual
            show_log=False,
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
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    """Extract raw text from an image or PDF. Returns lines with coordinates and confidence."""
    _verify_api_key(x_api_key)
    _check_rate_limit()

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    file_hash = hashlib.sha256(content).hexdigest()[:16]
    start = time.time()

    # Save to temp file (PaddleOCR needs file path)
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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

        return OCRResult(
            success=True,
            text="\n".join(full_text_parts),
            lines=lines,
            confidence=round(avg_confidence, 4),
            processing_time_ms=elapsed_ms,
            file_hash=file_hash,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    finally:
        import os
        os.unlink(tmp_path)


@app.post("/ocr/invoice", response_model=InvoiceFields)
async def extract_invoice(
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    """Extract structured invoice fields from an image/PDF. Uses OCR + post-processing."""
    _verify_api_key(x_api_key)
    _check_rate_limit()

    # First get raw text
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    start = time.time()

    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        ocr = _get_ocr()
        results = ocr.ocr(tmp_path, cls=True)

        raw_text = ""
        if results and results[0]:
            raw_text = "\n".join(line[1][0] for line in results[0])

        # Post-process: extract structured fields from raw text
        fields = _extract_invoice_fields(raw_text)
        elapsed_ms = int((time.time() - start) * 1000)

        return InvoiceFields(
            success=True,
            raw_text=raw_text,
            processing_time_ms=elapsed_ms,
            **fields,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invoice extraction failed: {str(e)}")

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


# ─── Startup ─────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    """Pre-load OCR model on startup."""
    print(f"OCR Server starting...")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"Rate limit: {MAX_REQUESTS_PER_MINUTE}/min")
    try:
        _get_ocr()
        print("PaddleOCR model loaded successfully")
    except Exception as e:
        print(f"WARNING: PaddleOCR not available: {e}")
        print("Server will start but OCR endpoints will fail")
