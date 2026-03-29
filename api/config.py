"""OCR Server Configuration."""

import hmac
import logging

from pydantic import model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger("optii")


class Settings(BaseSettings):
    # OCR
    max_file_size_mb: int = 20
    supported_languages: list[str] = ["fr", "nl", "de", "en"]

    # Auth (chut.auth)
    auth_url: str = "https://auth.chut.me"
    auth_client_id: str = ""
    auth_client_secret: str = ""

    # Database (Scaleway Serverless SQL)
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = ""
    db_password: str = ""
    db_name: str = "optii"
    db_pool_min: int = 2
    db_pool_max: int = 10
    db_ssl: bool = True

    # Internal API keys for direct access (chut.app, accounting.chut.app)
    # Comma-separated id:key pairs, e.g. "chut-app:key1,accounting:key2"
    internal_api_keys: str = ""

    # Rate limiting
    rate_limit_rpm: int = 60

    # Demo mode
    demo_enabled: bool = True
    demo_rate_limit_rpm: int = 10
    demo_max_file_size_mb: int = 5

    # CORS
    cors_origins: str = "https://optii.eu,https://www.optii.eu"

    model_config = {"env_prefix": "OPTII_"}

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        if not self.auth_client_id:
            logger.warning("OPTII_AUTH_CLIENT_ID is empty — Chut Auth login is disabled.")
        return self

    def get_internal_keys(self) -> dict[str, str]:
        """Parse internal API keys from env var."""
        keys: dict[str, str] = {}
        if not self.internal_api_keys:
            return keys
        for entry in self.internal_api_keys.split(","):
            entry = entry.strip()
            if ":" in entry:
                key_id, key_value = entry.split(":", 1)
                keys[key_id.strip()] = key_value.strip()
        return keys

    def is_internal_key(self, token: str) -> str | None:
        """Check if token matches an internal key. Returns key_id or None."""
        for key_id, key_value in self.get_internal_keys().items():
            if hmac.compare_digest(token, key_value):
                return key_id
        return None


settings = Settings()
