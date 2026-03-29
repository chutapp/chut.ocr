"""Pydantic models for auth, API keys, usage, and OCR."""

from __future__ import annotations

from pydantic import BaseModel, Field


# -- Tier configuration --

TIER_LIMITS: dict[str, dict] = {
    "free": {
        "max_rpm": 10,
        "max_file_size_mb": 5,
        "max_keys": 1,
        "monthly_quota": 500,
    },
    "professional": {
        "max_rpm": 60,
        "max_file_size_mb": 20,
        "max_keys": 5,
        "monthly_quota": 10_000,
    },
    "enterprise": {
        "max_rpm": 300,
        "max_file_size_mb": 20,
        "max_keys": 100,
        "monthly_quota": 0,  # unlimited
    },
}


# -- Auth --

class AuthResponse(BaseModel):
    user_id: str
    email: str
    tier: str


class UserProfile(BaseModel):
    id: str
    email: str
    tier: str
    created_at: str
    key_count: int


# -- API Keys --

class CreateKeyRequest(BaseModel):
    name: str = Field(default="", max_length=64)


class CreateKeyResponse(BaseModel):
    key: str
    key_prefix: str
    name: str
    created_at: str


class APIKeyInfo(BaseModel):
    id: str
    key_prefix: str
    name: str
    is_active: bool
    created_at: str
    last_used_at: str | None


# -- Usage --

class DailyUsage(BaseModel):
    date: str
    requests: int
    file_size_kb: int
    avg_ms: float


class UsageStats(BaseModel):
    period: str
    total_requests: int
    total_file_size_kb: int
    daily: list[DailyUsage]


class DashboardOverview(BaseModel):
    tier: str
    tier_limits: dict
    key_count: int
    total_requests: int
    total_file_size_kb: int


# -- OCR --

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
