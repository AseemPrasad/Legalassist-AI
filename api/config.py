"""
API Configuration
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


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
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
    ]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # requests
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_BURST: int = 200  # max burst
    RATE_LIMIT_ABUSE_THRESHOLD: int = 3  # consecutive denials before a temporary block
    RATE_LIMIT_ABUSE_WINDOW: int = 60  # seconds for abuse counter window
    RATE_LIMIT_ABUSE_BLOCK_SECONDS: int = 300  # block duration after abuse threshold
    
    # Authentication
    AUTH_ENABLED: bool = True
    # Prefer externally managed secrets; fall back to environment for local dev
    try:
        from utils.secret_manager import get_secret
        _jwt_from_vault = get_secret("jwt_secret")
    except Exception:
        _jwt_from_vault = None

    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", _jwt_from_vault or "")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    API_KEY_HEADER: str = "X-API-Key"
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost:5432/legalassist"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    CELERY_TASK_TIMEOUT: int = 3600  # 1 hour
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300  # 55 minutes
    
    # File Upload
    UPLOAD_MAX_SIZE: int = 500 * 1024 * 1024  # 500 MB
    UPLOAD_EXTENSIONS: list = [".pdf", ".doc", ".docx", ".txt", ".html"]
    UPLOAD_TEMP_DIR: str = "/tmp/legalassist-uploads"
    
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
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> APISettings:
    """Get API settings"""
    return APISettings()
