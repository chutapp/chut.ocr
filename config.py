"""OCR Server Configuration."""

import os
import secrets

# API key for authentication — set via environment variable or generate random
API_KEY = os.environ.get("OCR_API_KEY", secrets.token_urlsafe(32))

# Model config
MODEL_NAME = "PaddlePaddle/PaddleOCR"
SUPPORTED_LANGUAGES = ["fr", "nl", "de", "en"]
MAX_FILE_SIZE_MB = 20

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60

# Server
HOST = "0.0.0.0"
PORT = 8090
