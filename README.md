# chut.ocr

Secure document text extraction API. Runs PaddleOCR on GPU.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/ocr/extract` | API key | Raw text extraction (lines + confidence + bbox) |
| POST | `/ocr/invoice` | API key | Structured invoice field extraction |

## Usage

```bash
# Health check
curl https://ocr.chut.me/health

# Extract text from image
curl -X POST https://ocr.chut.me/ocr/extract \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@invoice.png"

# Extract invoice fields
curl -X POST https://ocr.chut.me/ocr/invoice \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@invoice.png"
```

## Response (invoice)

```json
{
  "success": true,
  "supplier_name": "Proximus NV",
  "supplier_vat": "BE0202239951",
  "invoice_date": "15/03/2026",
  "total_excl_vat": "75.00",
  "total_incl_vat": "90.75",
  "vat_amount": "15.75",
  "processing_time_ms": 89
}
```

## Stack

- **Model:** PaddleOCR 2.9 + PaddlePaddle GPU 2.6
- **GPU:** NVIDIA RTX 4000 SFF Ada (20GB VRAM)
- **Server:** FastAPI + uvicorn + nginx + Let's Encrypt
- **Security:** API key auth, rate limiting, fail2ban, UFW firewall

## Deploy

Push to `main` → GitHub Actions runs tests → deploys to GPU server via SSH.

Set these GitHub secrets:
- `OCR_SERVER_HOST`
- `OCR_SERVER_USER`
- `OCR_SERVER_SSH_KEY`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_API_KEY` | random | API authentication key |
| `OCR_RATE_LIMIT` | 60 | Max requests per minute |
| `MAX_FILE_SIZE_MB` | 20 | Max upload size |
# Version 1.0.1 — CI/CD test
