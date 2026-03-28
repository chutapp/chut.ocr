"""OCR Server Configuration."""

import os
import secrets

# API keys — supports rotation with multiple active keys.
# Set OCR_API_KEY for a single key, or OCR_API_KEYS for multiple (comma-separated id:key pairs).
# Examples:
#   OCR_API_KEY=my-secret-key                          -> single key, id="default"
#   OCR_API_KEYS=chut-app:key1,accounting:key2         -> multiple named keys
#
# To rotate: add the new key, deploy, update clients, remove the old key, deploy again.
_keys_env = os.environ.get("OCR_API_KEYS", "")
if _keys_env:
    API_KEYS: dict[str, str] = {}
    for entry in _keys_env.split(","):
        entry = entry.strip()
        if ":" in entry:
            key_id, key_value = entry.split(":", 1)
            API_KEYS[key_id.strip()] = key_value.strip()
else:
    _single_key = os.environ.get("OCR_API_KEY", secrets.token_urlsafe(32))
    API_KEYS = {"default": _single_key}

# Model config
MODEL_NAME = "PaddlePaddle/PaddleOCR"
SUPPORTED_LANGUAGES = ["fr", "nl", "de", "en"]
MAX_FILE_SIZE_MB = 20

# Rate limiting
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("OCR_RATE_LIMIT", "60"))

# CORS — allowed origins (comma-separated via env, or restrictive default)
_DEFAULT_ORIGINS = "https://chut.app,https://accounting.chut.app,https://optii.eu"
_origins_env = os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS)
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

# Server
HOST = "0.0.0.0"
PORT = 8090
