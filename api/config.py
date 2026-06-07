import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
import re

SENSITIVE_CONFIG_KEYS: set = {
    "JWT_SECRET_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "API_KEY_HEADER",
    "LLM_MODEL",
}


def _should_mask(key: str, value: Any) -> bool:
    """Return True if *value* should be masked in diagnostic output.

    Numeric values (ports, timeouts, limits, counts) are preserved since
    they carry operational context.  Only string values whose key matches
    a known sensitive pattern are masked.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return False
    key_lower = key.lower()
    if any(kw in key_lower for kw in ("_url", "_header")):
        return True
    return key in SENSITIVE_CONFIG_KEYS

# Initialize logging for config phase
logger = logging.getLogger(__name__)

# Load .env file
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"

if PROJECT_ENV_PATH.exists():
    load_dotenv(dotenv_path=PROJECT_ENV_PATH)
else:
    load_dotenv()

def _get_val(key, default=None):
    # Try Streamlit secrets first (if in a Streamlit context)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, RuntimeError, AttributeError, FileNotFoundError):
        pass
    
    # Fallback to environment variables
    return os.getenv(key, default)

def _get_bool_env(key, default=False):
    val = str(_get_val(key, str(default))).lower()
    return val in ("1", "true", "yes", "on")

def _get_int_env(key, default):
    try:
        return int(_get_val(key, str(default)))
    except (ValueError, TypeError):
        return default

class Config:
    # --- App Identity ---
    APP_NAME = _get_val("APP_NAME", "LegalEase AI")
    APP_ENV = _get_val("APP_ENV", _get_val("ENVIRONMENT", "development")).lower()
    DEBUG = _get_bool_env("DEBUG", APP_ENV in ("dev", "development", "local"))
    TESTING = _get_bool_env("TESTING", False)
    
    # --- Logging ---
    LOG_LEVEL = _get_val("LOG_LEVEL", "INFO")
    
    # --- Model Settings (LLM) ---
    # The primary model used for generating summaries and legal remedies analysis.
    # Default is Llama 3.1 8B Instruct via OpenRouter.
    DEFAULT_MODEL = _get_val("DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct")
    
    # Base URL for the OpenAI-compatible API (OpenRouter is used by default).
    OPENROUTER_BASE_URL = _get_val("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # API Key for OpenRouter. Must be provided for the AI features to work.
    OPENROUTER_API_KEY = _get_val("OPENROUTER_API_KEY", "")
    
    # --- AI Request Performance & Reliability ---
    # The maximum number of tokens allowed for judgment summaries.
    SUMMARY_MAX_TOKENS = _get_int_env("SUMMARY_MAX_TOKENS", 280)
    
    # The maximum number of tokens allowed for legal remedies analysis.
    REMEDIES_MAX_TOKENS = _get_int_env("REMEDIES_MAX_TOKENS", 900)
    
    # Controls the randomness of the AI output. 
    # Lower values (0.0-0.2) make the output more deterministic and focused.
    LLM_TEMPERATURE = float(_get_val("LLM_TEMPERATURE", "0.05"))
    
    # The timeout in seconds for AI model API requests. 
    # This is critical for preventing the application from hanging on slow network calls.
    AI_REQUEST_TIMEOUT = float(_get_val("AI_REQUEST_TIMEOUT", _get_val("LLM_TIMEOUT", "60.0")))
    
    # Alias for backward compatibility with legacy code.
    LLM_TIMEOUT = AI_REQUEST_TIMEOUT 
    
    # The maximum number of retry attempts for failed AI requests (e.g., on rate limits).
    AI_MAX_RETRIES = _get_int_env("AI_MAX_RETRIES", 3)
    
    # The base delay in seconds for exponential backoff during retries.
    AI_RETRY_BACKOFF_BASE = float(_get_val("AI_RETRY_BACKOFF_BASE", "2.0"))

    # --- OCR Settings ---
    OCR_ENABLED = _get_bool_env("OCR_ENABLED", False)
    OCR_LANGUAGES = _get_val("OCR_LANGUAGES", "eng+hin")
    OCR_DPI = _get_int_env("OCR_DPI", 300)
    
    # --- File Processing ---
    MAX_FILE_SIZE_MB = _get_int_env("MAX_FILE_SIZE_MB", 25)
    WARN_FILE_SIZE_MB = _get_int_env("WARN_FILE_SIZE_MB", 10)
    TEXT_COMPRESSION_LIMIT = _get_int_env("TEXT_COMPRESSION_LIMIT", 6000)
    # --- Attachments ---
    # Directory where uploaded attachments are stored (development)
    ATTACHMENTS_DIR = _get_val("ATTACHMENTS_DIR", str(PROJECT_ROOT / "attachments"))
    # Use randomized filenames to avoid collisions and leaking original names
    ATTACHMENTS_RANDOMIZE_FILENAMES = _get_bool_env("ATTACHMENTS_RANDOMIZE_FILENAMES", True)
    
    # --- Database Settings ---
    DATABASE_URL = _get_val("DATABASE_URL", "sqlite:///./legalassist.db")

    # --- Backend API Settings ---
    API_BASE_URL = _get_val("API_BASE_URL", "")
    API_REQUEST_TIMEOUT_SECONDS = float(_get_val("API_REQUEST_TIMEOUT_SECONDS", "5.0"))
    
    # --- Celery Settings ---
    CELERY_TASK_TIMEOUT = _get_int_env("CELERY_TASK_TIMEOUT", 3600)
    CELERY_TASK_SOFT_TIME_LIMIT = _get_int_env("CELERY_TASK_SOFT_TIME_LIMIT", 3300)
    
    # --- WebSocket Settings ---
    ENABLE_WEBSOCKET = _get_bool_env("ENABLE_WEBSOCKET", True)
    WEBSOCKET_RATE_LIMIT_REQUESTS = _get_int_env("WEBSOCKET_RATE_LIMIT_REQUESTS", 30)
    WEBSOCKET_RATE_LIMIT_WINDOW = _get_int_env("WEBSOCKET_RATE_LIMIT_WINDOW", 60)
    ALLOWED_HOSTS = ["localhost", "127.0.0.1", "*.example.com"]
    
    # --- Authentication (JWT & OTP) ---
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = _get_int_env("JWT_EXPIRY_HOURS", 7 * 24)
    OTP_EXPIRY_MINUTES = _get_int_env("OTP_EXPIRY_MINUTES", 10)
    OTP_MAX_ATTEMPTS = _get_int_env("OTP_MAX_ATTEMPTS", 3)
    JWT_ISSUER = _get_val("JWT_ISSUER", "legalassist.ai")
    JWT_AUDIENCE = _get_val("JWT_AUDIENCE", "legalassist-users")
    
    @classmethod
    def get_jwt_secret(cls):
        """
        Resolve JWT secret securely.
        
        JWT_SECRET must be provided via environment variable or Streamlit secrets.
        File-based secrets are no longer supported for security.
        
        Raises:
            RuntimeError: If JWT_SECRET is not configured in environment variables.
        """
        # Try environment / streamlit secrets first
        secret = str(_get_val("JWT_SECRET", "")).strip()
        if secret:
            return secret

        # Try central secret manager (Vault or env fallback)
        try:
            from utils.secret_manager import get_secret
            vault_secret = get_secret("jwt_secret")
            if vault_secret:
                return str(vault_secret).strip()
        except Exception:
            pass

        env_name = cls.APP_ENV.upper()
        raise RuntimeError(
            f"JWT_SECRET is not configured for the {env_name} environment. "
            "For security, secrets must be explicitly provided via the 'JWT_SECRET' "
            "environment variable or a configured Vault."
        )

    # --- Notification Settings (SMS) ---
    TWILIO_ACCOUNT_SID = _get_val("TWILIO_ACCOUNT_SID", "")
    TWILIO_FROM_NUMBER = _get_val("TWILIO_FROM_NUMBER", "")
    TWILIO_AUTH_TOKEN = None

    @classmethod
    def get_twilio_auth_token(cls) -> str:
        """Return the Twilio auth token, retrieved on demand to limit exposure."""
        if cls.TWILIO_AUTH_TOKEN is not None:
            return cls.TWILIO_AUTH_TOKEN
        # Prefer centralized secret manager
        try:
            from utils.secret_manager import get_secret
            val = get_secret("twilio_auth_token") or _get_val("TWILIO_AUTH_TOKEN", "")
            return str(val or "")
        except Exception:
            return str(_get_val("TWILIO_AUTH_TOKEN", "") or "")

    # --- Notification Settings (Email) ---
    SENDGRID_FROM_EMAIL = _get_val("SENDGRID_FROM_EMAIL", "noreply@legalassist.ai")
    SENDGRID_API_KEY = None

    @classmethod
    def get_sendgrid_api_key(cls) -> str:
        """Return the SendGrid API key, retrieved on demand to limit exposure."""
        if cls.SENDGRID_API_KEY is not None:
            return cls.SENDGRID_API_KEY
        try:
            from utils.secret_manager import get_secret
            val = get_secret("sendgrid_api_key") or _get_val("SENDGRID_API_KEY", "")
            return str(val or "")
        except Exception:
            return str(_get_val("SENDGRID_API_KEY", "") or "")

    # --- Rate Limiting ---
    RATE_LIMIT_ENABLED = _get_bool_env("RATE_LIMIT_ENABLED", False)
    REDIS_URL = _get_val("REDIS_URL", "")
    RATE_LIMIT_REQUESTS = _get_int_env("RATE_LIMIT_REQUESTS", 100)
    RATE_LIMIT_WINDOW = _get_int_env("RATE_LIMIT_WINDOW", 60)
    RATE_LIMIT_BURST = _get_int_env("RATE_LIMIT_BURST", 10)
    RATE_LIMIT_ABUSE_THRESHOLD = _get_int_env("RATE_LIMIT_ABUSE_THRESHOLD", 3)
    RATE_LIMIT_ABUSE_WINDOW = _get_int_env("RATE_LIMIT_ABUSE_WINDOW", 60)
    RATE_LIMIT_ABUSE_BLOCK_SECONDS = _get_int_env("RATE_LIMIT_ABUSE_BLOCK_SECONDS", 300)
    AUTH_RATE_LIMIT_REQUESTS = _get_int_env("AUTH_RATE_LIMIT_REQUESTS", 5)
    AUTH_RATE_LIMIT_WINDOW = _get_int_env("AUTH_RATE_LIMIT_WINDOW", 60)
    API_KEY_HEADER = _get_val("API_KEY_HEADER", "X-API-Key")

    # --- CORS / API Server ---
    CORS_ORIGINS = _get_val("CORS_ORIGINS", "http://localhost:8080")
    API_TITLE = _get_val("API_TITLE", "LegalAssist API")
    API_VERSION = _get_val("API_VERSION", "1.0.0")
    API_HOST = _get_val("API_HOST", "0.0.0.0")
    API_PORT = _get_int_env("API_PORT", 8000)
    API_WORKERS = _get_int_env("API_WORKERS", 1)
    ENABLE_WEBSOCKET = _get_bool_env("ENABLE_WEBSOCKET", False)

    # --- Application URLs ---
    BASE_URL = _get_val("BASE_URL", "https://legalassist.ai")

    @classmethod
    def is_development(cls):
        env_dev = cls.APP_ENV in ("dev", "development", "local") or cls.DEBUG or cls.TESTING
        if not env_dev:
            return False
        # Secondary safety check: flag if BASE_URL looks like production
        base = str(cls.BASE_URL or "").lower()
        if not any(local in base for local in ("localhost", "127.0.0.1", "0.0.0.0", "::1")):
            import logging
            logging.getLogger(__name__).warning(
                "is_development()=True but BASE_URL=%s suggests a non-local deployment. "
                "Review APP_ENV / DEBUG / TESTING settings.",
                base,
            )
        return env_dev

    @classmethod
    def is_production(cls):
        return cls.APP_ENV in ("production", "prod", "live")


import re
import copy

class ConfigSanitizer:
    """
    Utility class to sanitize configuration dictionaries before logging to prevent
    accidental leakage of sensitive credentials like API keys and JWT secrets.
    """
    
    # Patterns for keys that likely contain sensitive data
    SENSITIVE_KEY_PATTERNS = [
        re.compile(r"secret", re.IGNORECASE),
        re.compile(r"key", re.IGNORECASE),
        re.compile(r"token", re.IGNORECASE),
        re.compile(r"password", re.IGNORECASE),
        re.compile(r"pwd", re.IGNORECASE),
        re.compile(r"auth", re.IGNORECASE),
        re.compile(r"credential", re.IGNORECASE),
        re.compile(r"jwt", re.IGNORECASE),
        re.compile(r"sid", re.IGNORECASE),
        re.compile(r"api_?key", re.IGNORECASE),
        re.compile(r"access", re.IGNORECASE),
        re.compile(r"url", re.IGNORECASE),
    ]

    # Keys that we explicitly want to mask
    EXPLICIT_SENSITIVE_KEYS = {
        "DATABASE_URL",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "JWT_SECRET",
        "JWT_SECRET_PREVIOUS",
        "JWT_SECRET_KEY",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "SENDGRID_API_KEY",
    }
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # requests per endpoint window
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_BURST: int = 200  # max burst
    GLOBAL_RATE_LIMIT_REQUESTS: int = 200  # requests across all endpoints
    GLOBAL_RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Authentication
    AUTH_ENABLED: bool = True
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
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

    def sanitized_dict(self) -> Dict[str, Any]:
        """Return settings as a dict with sensitive values masked.

        Numeric operational values (ports, timeouts, limits, counts) are
        preserved so they remain useful for diagnostics.  Only string values
        from sensitive keys are replaced with a placeholder.
        """
        raw = self.model_dump()
        result: Dict[str, Any] = {}
        for key, value in raw.items():
            if _should_mask(key, value):
                result[key] = "***"
            else:
                result[key] = value
        return result


def get_settings() -> APISettings:
    """Get API settings"""
    return APISettings()
