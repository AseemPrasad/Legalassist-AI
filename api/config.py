
import tempfile
"""
API Configuration
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API Configuration"""
    
    # API Info
    API_TITLE: str = "Legalassist-AI"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Comprehensive legal case analysis and deadline management API"
    
    # Server
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    REQUIRE_HTTPS: bool = os.getenv("REQUIRE_HTTPS", "true").lower() == "true"
    TRUSTED_HOSTS: list = [
        "localhost",
        "127.0.0.1",
        "::1",
    ]
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
    ]
    
    # Allowed Hosts for TrustedHostMiddleware
    # MUST be set via APP_ALLOWED_HOSTS environment variable in production
    # Format: comma-separated (localhost,127.0.0.1,example.com) or JSON array
    ALLOWED_HOSTS: list = []
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # requests
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_BURST: int = 200  # max burst
    
    # Authentication Rate Limiting (Credential Stuffing Protection)
    AUTH_RATE_LIMIT_REQUESTS: int = 5  # tight limit for login
    AUTH_RATE_LIMIT_WINDOW: int = 60   # per minute
    AUTH_RATE_LIMIT_STRATEGY: str = "fixed-window"  # or 'sliding-window'

    # WebSocket Rate Limiting
    WEBSOCKET_RATE_LIMIT_REQUESTS: int = 20
    WEBSOCKET_RATE_LIMIT_WINDOW: int = 60
    
    # Authentication
    AUTH_ENABLED: bool = True
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET", os.getenv("JWT_SECRET_KEY", ""))
    JWT_SECRET_KEY_PREVIOUS: str = os.getenv("JWT_SECRET_PREVIOUS", os.getenv("JWT_SECRET_KEY_PREVIOUS", ""))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_MINUTES: int = 15
    JWT_EXPIRATION_HOURS: int = 24
    JWT_ISSUER: str = os.getenv("JWT_ISSUER", "legalassist.ai")
    JWT_AUDIENCE: str = os.getenv("JWT_AUDIENCE", "legalassist-users")
    API_KEY_HEADER: str = "X-API-Key"
    CSRF_SECRET: str = ""
    
    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if not v or v == "your-secret-key-change-in-production":
            raise ValueError(
                "JWT_SECRET (or JWT_SECRET_KEY) must be set to a secure value. "
                "Do not use default or placeholder values in production."
            )
        return v
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis — validated lazily in __init__; empty default so APISettings can be imported
    REDIS_URL: str = ""
    REDIS_CACHE_TTL: int = 3600  # 1 hour

    # Celery — validated lazily in __init__; empty default so APISettings can be imported
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""
    CELERY_TASK_TIMEOUT: int = 3600  # 1 hour
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300  # 55 minutes
    
    # File Upload
    UPLOAD_MAX_SIZE: int = 25 * 1024 * 1024  # 25 MB
    UPLOAD_EXTENSIONS: list = [".pdf", ".doc", ".docx", ".txt"]
    UPLOAD_TEMP_DIR: str = os.getenv(
        "UPLOAD_TEMP_DIR",
        str(Path(tempfile.gettempdir()) / "legalassist-uploads")
    )
    
    # PDF Export
    PDF_MAX_PAGES: int = 5000
    PDF_QUALITY: str = "high"  # low, medium, high
    
    # LLM Settings
    LLM_MAX_TOKENS: int = 2000
    LLM_TEMPERATURE: float = 0.7
    LLM_MODEL: str = "gpt-4"
    LLM_TIMEOUT: int = 120  # seconds
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "json"
    
    # Observability
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    JAEGER_ENABLED: bool = os.getenv("JAEGER_ENABLED", "false").lower() == "true"
    
    # Feature Flags
    ENABLE_OAUTH2: bool = os.getenv("ENABLE_OAUTH2", "true").lower() == "true"
    ENABLE_WEBSOCKET: bool = os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true"
    ENABLE_ANALYTICS: bool = os.getenv("ENABLE_ANALYTICS", "true").lower() == "true"
    
    def __init__(self, **data):
        # Enforce canonical env var precedence BEFORE Pydantic validates fields.
        # JWT_SECRET takes priority over the legacy JWT_SECRET_KEY alias.
        # Init kwargs have the highest priority in Pydantic v2 BaseSettings, so
        # injecting here ensures the field_validator sees the canonical value.
        if not data.get("JWT_SECRET_KEY"):
            canonical = os.getenv("JWT_SECRET") or os.getenv("JWT_SECRET_KEY", "")
            if canonical:
                data["JWT_SECRET_KEY"] = canonical
        if not data.get("JWT_SECRET_KEY_PREVIOUS"):
            canonical_prev = os.getenv("JWT_SECRET_PREVIOUS") or os.getenv("JWT_SECRET_KEY_PREVIOUS", "")
            if canonical_prev:
                data["JWT_SECRET_KEY_PREVIOUS"] = canonical_prev
        # Resolve CSRF_SECRET from env (canonical config source for CSRF signing)
        if not data.get("CSRF_SECRET"):
            data["CSRF_SECRET"] = os.getenv("CSRF_SECRET", "")
        # Lazily resolve Redis & Celery URLs from env (fail only when instantiated)
        if not data.get("REDIS_URL"):
            data["REDIS_URL"] = os.getenv("REDIS_URL", "")
        if not data.get("CELERY_BROKER_URL"):
            data["CELERY_BROKER_URL"] = os.getenv("CELERY_BROKER_URL", "")
        if not data.get("CELERY_RESULT_BACKEND"):
            data["CELERY_RESULT_BACKEND"] = os.getenv("CELERY_RESULT_BACKEND", "")
        super().__init__(**data)
        # Fail hard if Redis/Celery env vars are missing in production
        is_prod = self.ENVIRONMENT in ("production", "prod", "live")
        if is_prod:
            if not self.REDIS_URL:
                raise ValueError("REDIS_URL environment variable is required in production.")
            if not self.CELERY_BROKER_URL:
                raise ValueError("CELERY_BROKER_URL environment variable is required in production.")
            if not self.CELERY_RESULT_BACKEND:
                raise ValueError("CELERY_RESULT_BACKEND environment variable is required in production.")
        # Parse ALLOWED_HOSTS from environment
        hosts_env = os.getenv("APP_ALLOWED_HOSTS", "")
        if hosts_env.strip():
            # Support both comma-separated and JSON formats
            if hosts_env.startswith('['):
                import json
                try:
                    self.ALLOWED_HOSTS = json.loads(hosts_env)
                except (json.JSONDecodeError, ValueError):
                    self.ALLOWED_HOSTS = [h.strip() for h in hosts_env.split(',') if h.strip()]
            else:
                self.ALLOWED_HOSTS = [h.strip() for h in hosts_env.split(',') if h.strip()]
        else:
            # Refuse to run without explicit host configuration
            raise ValueError(
                "APP_ALLOWED_HOSTS environment variable must be set. "
                "Cannot start with empty allowed hosts list for security."
            )
    
    def validate_runtime_security(self) -> None:
        """Fail fast when production settings are insecure or missing required secrets."""
        is_prod = self.ENVIRONMENT in ("production", "prod", "live")
        if is_prod:
            if not self.JWT_SECRET_KEY or self.JWT_SECRET_KEY == "your-secret-key-change-in-production":
                raise RuntimeError("JWT_SECRET_KEY must be set to a secure value in production")
        origins = self.CORS_ORIGINS
        if isinstance(origins, list) and "*" in origins:
            import logging as _log
            msg = (
                "CORS configured with wildcard origin '*' and allow_credentials=True. "
                "This allows any website to make credentialed requests. "
                "Set explicit origins in production."
            )
            if is_prod:
                raise RuntimeError(msg)
            _log.warning(msg)

        csrf_secret = self.CSRF_SECRET
        if not csrf_secret or len(csrf_secret) < 16:
            msg = (
                "CSRF_SECRET environment variable must be set to a secure value "
                "(minimum 16 characters). Without it, CSRF tokens are per-worker "
                "and cross-worker requests will fail."
            )
            if is_prod:
                raise RuntimeError(msg)
            _log.warning(msg)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


@lru_cache()
def get_settings() -> APISettings:
    """Get API settings (cached)"""
    return APISettings()
