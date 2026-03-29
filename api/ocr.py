"""OCR endpoints: text extraction + invoice parsing + demo."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import PurePosixPath

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from .config import settings
from .db import get_db
from .models import OCRResult, InvoiceFields

logger = logging.getLogger("optii")

router = APIRouter(tags=["ocr"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}

# ─── PaddleOCR engine ─────────────────────────────────────────────────────

_ocr_engine = None


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        import inspect

        params = inspect.signature(PaddleOCR.__init__).parameters
        kwargs: dict = {"lang": "fr"}

        if "use_textline_orientation" in params:
            kwargs["use_textline_orientation"] = True
        elif "use_angle_cls" in params:
            kwargs["use_angle_cls"] = True

        if "show_log" in params:
            kwargs["show_log"] = False

        _ocr_engine = PaddleOCR(**kwargs)
    return _ocr_engine


def load_model() -> None:
    """Pre-load OCR model. Called at startup."""
    _get_ocr()


# ─── OCR processing ──────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: str) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img_path = f"{pdf_path}_page{page_num}.png"
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def _ocr_single_image(image_path: str) -> list[dict]:
    ocr = _get_ocr()

    try:
        results = ocr.ocr(image_path, cls=True)
    except TypeError:
        results = ocr.ocr(image_path)

    lines = []
    if not results:
        return lines

    page = results[0] if isinstance(results, list) and results else results
    if not page:
        return lines

    for item in page:
        if isinstance(item, dict):
            lines.append({
                "text": item.get("text", item.get("rec_text", "")),
                "confidence": round(item.get("score", item.get("rec_score", 0.0)), 4),
                "bbox": item.get("dt_polys", item.get("bbox", [])),
            })
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            bbox = item[0]
            text_info = item[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                lines.append({
                    "text": text_info[0],
                    "confidence": round(text_info[1], 4),
                    "bbox": bbox,
                })

    return lines


def _run_ocr(file_path: str) -> list[dict]:
    if file_path.lower().endswith(".pdf"):
        image_paths = _pdf_to_images(file_path)
        all_lines = []
        try:
            for img_path in image_paths:
                all_lines.extend(_ocr_single_image(img_path))
        finally:
            for img_path in image_paths:
                try:
                    os.unlink(img_path)
                except OSError:
                    pass
        return all_lines
    else:
        return _ocr_single_image(file_path)


def _validate_file_extension(filename: str | None) -> str:
    safe_name = PurePosixPath(filename).name if filename else ""
    suffix = PurePosixPath(safe_name).suffix.lower() if safe_name else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return suffix


def _extract_invoice_fields(text: str) -> dict:
    fields = {}

    vat_match = re.search(r"(?:TVA|BTW|VAT|Numéro de TVA)[:\s]*([A-Z]{2}\d{8,10})", text)
    if vat_match:
        fields["supplier_vat"] = vat_match.group(1)

    inv_match = re.search(r"(?:Facture|Invoice|Factuur)\s*(?:N[°o]?|#)?[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if inv_match:
        fields["invoice_number"] = inv_match.group(1)

    date_match = re.search(r"(?:Date)[:\s]*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})", text, re.IGNORECASE)
    if not date_match:
        date_match = re.search(r"(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})", text, re.IGNORECASE)
    if date_match:
        fields["invoice_date"] = date_match.group(1)

    total_match = re.search(r"(?:Total\s*(?:TTC|incl|TVAC|à payer))[:\s]*(\d+[\s.,]\d{2})\s*(?:EUR|€)?", text, re.IGNORECASE)
    if total_match:
        fields["total_incl_vat"] = total_match.group(1).replace(",", ".").replace(" ", "")

    ht_match = re.search(r"(?:Total\s*(?:HT|excl|HTVA|hors))[:\s]*(\d+[\s.,]\d{2})\s*(?:EUR|€)?", text, re.IGNORECASE)
    if ht_match:
        fields["total_excl_vat"] = ht_match.group(1).replace(",", ".").replace(" ", "")

    tva_match = re.search(r"(?:TVA|BTW|VAT)\s*\d+%[:\s]*(\d+[\s.,]\d{2})\s*(?:EUR|€)?", text, re.IGNORECASE)
    if tva_match:
        fields["vat_amount"] = tva_match.group(1).replace(",", ".").replace(" ", "")

    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) > 3 and not line.upper().startswith(
            ("FACTURE", "INVOICE", "DATE", "TVA", "TOTAL", "NUMÉRO")
        ):
            fields["supplier_name"] = line
            break

    return fields


# ─── Usage logging ────────────────────────────────────────────────────────

async def _log_usage(api_key_id: str | None, endpoint: str, file_size_kb: int,
                     processing_ms: int, status_code: int) -> None:
    if not api_key_id:
        return
    try:
        pool = get_db()
        await pool.execute(
            "INSERT INTO usage_log (api_key_id, endpoint, file_size_kb, processing_ms, status_code, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            api_key_id, endpoint, file_size_kb, processing_ms, status_code,
            datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        pass


# ─── OCR processing helper ───────────────────────────────────────────────

async def _process_ocr(file: UploadFile, request: Request, max_size_mb: int) -> tuple[list[dict], int, str, int]:
    """Shared OCR processing. Returns (lines, elapsed_ms, file_hash, file_size_kb)."""
    suffix = _validate_file_extension(file.filename)

    content = await file.read()
    if len(content) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {max_size_mb}MB)")

    file_hash = hashlib.sha256(content).hexdigest()[:16]
    file_size_kb = len(content) // 1024
    start = time.time()

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(content)
        tmp.close()
        tmp_path = tmp.name
        parsed_lines = await run_in_threadpool(_run_ocr, tmp_path)
        elapsed_ms = int((time.time() - start) * 1000)
        return parsed_lines, elapsed_ms, file_hash, file_size_kb
    finally:
        os.unlink(tmp.name)


# ─── Endpoints ────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "PaddleOCR",
        "languages": settings.supported_languages,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/ocr/extract", response_model=OCRResult)
async def extract_text(request: Request, file: UploadFile = File(...)):
    """Extract raw text from an image or PDF."""
    max_size = getattr(request.state, "tier_limits", {}).get("max_file_size_mb", settings.max_file_size_mb)

    try:
        parsed_lines, elapsed_ms, file_hash, file_size_kb = await _process_ocr(file, request, max_size)
    except HTTPException:
        raise
    except Exception:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=500, detail="OCR processing failed")

    total_confidence = sum(l["confidence"] for l in parsed_lines)
    count = len(parsed_lines)
    avg_confidence = total_confidence / count if count > 0 else 0.0

    api_key_id = getattr(request.state, "api_key_id", None)
    asyncio.create_task(_log_usage(api_key_id, "/ocr/extract", file_size_kb, elapsed_ms, 200))

    return OCRResult(
        success=True,
        text="\n".join(l["text"] for l in parsed_lines),
        lines=parsed_lines,
        confidence=round(avg_confidence, 4),
        processing_time_ms=elapsed_ms,
        file_hash=file_hash,
    )


@router.post("/ocr/invoice", response_model=InvoiceFields)
async def extract_invoice(request: Request, file: UploadFile = File(...)):
    """Extract structured invoice fields from an image/PDF."""
    max_size = getattr(request.state, "tier_limits", {}).get("max_file_size_mb", settings.max_file_size_mb)

    try:
        parsed_lines, elapsed_ms, file_hash, file_size_kb = await _process_ocr(file, request, max_size)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Invoice extraction failed")
        raise HTTPException(status_code=500, detail="Invoice extraction failed")

    raw_text = "\n".join(l["text"] for l in parsed_lines)
    fields = _extract_invoice_fields(raw_text)
    avg_confidence = sum(l["confidence"] for l in parsed_lines) / len(parsed_lines) if parsed_lines else 0.0

    api_key_id = getattr(request.state, "api_key_id", None)
    asyncio.create_task(_log_usage(api_key_id, "/ocr/invoice", file_size_kb, elapsed_ms, 200))

    return InvoiceFields(
        success=True,
        raw_text=raw_text,
        confidence=round(avg_confidence, 4),
        processing_time_ms=elapsed_ms,
        **fields,
    )


@router.post("/demo/extract", response_model=OCRResult)
async def demo_extract(request: Request, file: UploadFile = File(...)):
    """Demo endpoint — rate-limited, smaller file size, no auth required."""
    try:
        parsed_lines, elapsed_ms, file_hash, file_size_kb = await _process_ocr(
            file, request, settings.demo_max_file_size_mb,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Demo OCR processing failed")
        raise HTTPException(status_code=500, detail="OCR processing failed")

    total_confidence = sum(l["confidence"] for l in parsed_lines)
    count = len(parsed_lines)
    avg_confidence = total_confidence / count if count > 0 else 0.0

    return OCRResult(
        success=True,
        text="\n".join(l["text"] for l in parsed_lines),
        lines=parsed_lines,
        confidence=round(avg_confidence, 4),
        processing_time_ms=elapsed_ms,
        file_hash=file_hash,
    )


@router.post("/demo/invoice", response_model=InvoiceFields)
async def demo_invoice(request: Request, file: UploadFile = File(...)):
    """Demo invoice endpoint — rate-limited, no auth required."""
    try:
        parsed_lines, elapsed_ms, file_hash, file_size_kb = await _process_ocr(
            file, request, settings.demo_max_file_size_mb,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Demo invoice extraction failed")
        raise HTTPException(status_code=500, detail="Invoice extraction failed")

    raw_text = "\n".join(l["text"] for l in parsed_lines)
    fields = _extract_invoice_fields(raw_text)
    avg_confidence = sum(l["confidence"] for l in parsed_lines) / len(parsed_lines) if parsed_lines else 0.0

    return InvoiceFields(
        success=True,
        raw_text=raw_text,
        confidence=round(avg_confidence, 4),
        processing_time_ms=elapsed_ms,
        **fields,
    )
