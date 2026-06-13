import os
import logging
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Initialize logging for config phase
logger = logging.getLogger(__name__)

# Load .env file
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"

if PROJECT_ENV_PATH.exists():
    load_dotenv(dotenv_path=PROJECT_ENV_PATH)
else:
    load_dotenv()

_STREAMLIT_SECRETS = None

def _load_streamlit_secrets():
    """Load Streamlit secrets once and cache them."""
    global _STREAMLIT_SECRETS
    if _STREAMLIT_SECRETS is not None:
        return _STREAMLIT_SECRETS
    try:
        import streamlit as st
        _STREAMLIT_SECRETS = dict(st.secrets)
    except (ImportError, RuntimeError, AttributeError, FileNotFoundError):
        _STREAMLIT_SECRETS = {}
    return _STREAMLIT_SECRETS


def _get_val(key, default=None):
    secrets = _load_streamlit_secrets()
    if key in secrets:
        return secrets[key]
    return os.getenv(key, default)

def _get_bool_env(key, default=False):
    val = str(_get_val(key, str(default))).lower()
    return val in ("1", "true", "yes", "on")

def _get_int_env(key, default):
    try:
        return int(_get_val(key, str(default)))
    except (ValueError, TypeError):
        return default

class _ConfigMeta(type):
    """Metaclass that resolves debug-related config dynamically from environment."""

    _DYNAMIC = frozenset({"APP_ENV", "DEBUG", "TESTING", "LOG_LEVEL"})

    def __getattr__(cls, name):
        if name == "APP_ENV":
            return _get_val("APP_ENV", _get_val("ENVIRONMENT", "development")).lower()
        if name == "DEBUG":
            return _get_bool_env("DEBUG", cls.APP_ENV in ("dev", "development", "local"))
        if name == "TESTING":
            return _get_bool_env("TESTING", False)
        if name == "LOG_LEVEL":
            return _get_val("LOG_LEVEL", "INFO")
        raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")


class Config(metaclass=_ConfigMeta):
    # --- App Identity ---
    APP_NAME = _get_val("APP_NAME", "LegalEase AI")
    
    # APP_ENV, DEBUG, TESTING, LOG_LEVEL are resolved dynamically via _ConfigMeta.__getattr__
    
    # --- Model Settings (LLM) ---
    # The primary model used for generating summaries and legal remedies analysis.
    # Default is Llama 3.1 8B Instruct via OpenRouter.
    DEFAULT_MODEL = _get_val("DEFAULT_MODEL", "meta-llama/llama-3.1-8b-instruct")
    
    # Base URL for the OpenAI-compatible API (OpenRouter is used by default).
    OPENROUTER_BASE_URL = _get_val("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    # API Key for OpenRouter. Must be provided for the AI features to work.
    OPENROUTER_API_KEY = _get_val("OPENROUTER_API_KEY", "")
    
    # --- Fallback/Secondary Model Settings ---
    SECONDARY_MODEL = _get_val("SECONDARY_MODEL", "gpt-4o-mini")
    SECONDARY_BASE_URL = _get_val("SECONDARY_BASE_URL", "https://api.openai.com/v1")
    SECONDARY_API_KEY = _get_val("SECONDARY_API_KEY", "")
    
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
    
    # --- Timezone Strategy ---
    USE_CLOCK_UTILITY = _get_bool_env("USE_CLOCK_UTILITY", True)
    DEFAULT_TIMEZONE = _get_val("DEFAULT_TIMEZONE", "UTC")
    
    # The maximum number of retry attempts for failed AI requests (e.g., on rate limits).
    AI_MAX_RETRIES = _get_int_env("AI_MAX_RETRIES", 3)
    
    # The base delay in seconds for exponential backoff during retries.
    AI_RETRY_BACKOFF_BASE = float(_get_val("AI_RETRY_BACKOFF_BASE", "2.0"))

    # --- OCR Settings ---
    OCR_ENABLED = _get_bool_env("OCR_ENABLED", False)
    OCR_LANGUAGES = _get_val("OCR_LANGUAGES", "eng+hin")
    OCR_DPI = _get_int_env("OCR_DPI", 300)
    OCR_CONFIDENCE_THRESHOLD = float(_get_val("OCR_CONFIDENCE_THRESHOLD", "0.5"))
    CLOUD_OCR_API_KEY = _get_val("CLOUD_OCR_API_KEY", "")
    CLOUD_OCR_ENDPOINT = _get_val("CLOUD_OCR_ENDPOINT", "")
    
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
    
    # --- Authentication (JWT & OTP) ---
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = _get_int_env("JWT_EXPIRY_HOURS", 7 * 24)
    OTP_EXPIRY_MINUTES = _get_int_env("OTP_EXPIRY_MINUTES", 10)
    OTP_MAX_ATTEMPTS = _get_int_env("OTP_MAX_ATTEMPTS", 3)
    OTP_RATE_LIMIT_HOURS = _get_int_env("OTP_RATE_LIMIT_HOURS", 1)
    OTP_RATE_LIMIT_MAX = _get_int_env("OTP_RATE_LIMIT_MAX", 3)
    
    _MISSING = object()
    _cached_jwt_secret = _MISSING

    @classmethod
    def get_jwt_secret(cls):
        """
        Resolve JWT secret securely.
        
        JWT_SECRET must be provided via environment variable or Streamlit secrets.
        File-based secrets are no longer supported for security.
        
        Raises:
            RuntimeError: If JWT_SECRET is not configured in environment variables.
        """
        if cls._cached_jwt_secret is not cls._MISSING:
            return cls._cached_jwt_secret

        secret = str(_get_val("JWT_SECRET", "")).strip()
        if secret:
            cls._cached_jwt_secret = secret
            return secret
            
        secret_file = PROJECT_ROOT / ".jwt_secret"
        if secret_file.exists():
            try:
                file_secret = secret_file.read_text(encoding="utf-8").strip()
                if file_secret:
                    cls._cached_jwt_secret = file_secret
                    return file_secret
            except Exception as e:
                logger.warning(f"Failed to read .jwt_secret file: {e}")
        
        # We no longer auto-generate secrets to prevent insecure fallback.
        # This forces explicit configuration which is a security best practice.
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

    _cached_twilio_auth_token = _MISSING

    @classmethod
    def get_twilio_auth_token(cls) -> str:
        """Return the Twilio auth token, retrieved on demand to limit exposure."""
        if cls._cached_twilio_auth_token is not cls._MISSING:
            return cls._cached_twilio_auth_token
        val = str(_get_val("TWILIO_AUTH_TOKEN", "") or "")
        cls._cached_twilio_auth_token = val
        return val

    # --- Notification Settings (Email) ---
    SENDGRID_FROM_EMAIL = _get_val("SENDGRID_FROM_EMAIL", "noreply@legalassist.ai")
    SENDGRID_API_KEY = None

    _cached_sendgrid_api_key = _MISSING

    @classmethod
    def get_sendgrid_api_key(cls) -> str:
        """Return the SendGrid API key, retrieved on demand to limit exposure."""
        if cls._cached_sendgrid_api_key is not cls._MISSING:
            return cls._cached_sendgrid_api_key
        val = str(_get_val("SENDGRID_API_KEY", "") or "")
        cls._cached_sendgrid_api_key = val
        return val

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
    
    @classmethod
    def is_sensitive(cls, key: str) -> bool:
        """
        Determine if a configuration key represents sensitive information.
        """
        if key.upper() in cls.EXPLICIT_SENSITIVE_KEYS:
            return True
            
        for pattern in cls.SENSITIVE_KEY_PATTERNS:
            if pattern.search(key):
                return True
                
        return False
        
    @classmethod
    def mask_string(cls, value: str) -> str:
        """
        Mask a string value.
        - Strings <= 4 chars: completely masked with asterisks
        - Strings 5-8 chars: show 1st char, mask rest, show last char
        - Strings > 8 chars: show 1st 2 chars, mask rest, show last 2 chars
        """
        if not value:
            return value
            
        length = len(value)
        if length <= 4:
            return "*" * length
        elif length <= 8:
            return f"{value[0]}{'*' * (length - 2)}{value[-1]}"
        else:
            return f"{value[:2]}{'*' * (length - 4)}{value[-2:]}"

    @classmethod
    def sanitize_value(cls, value: any) -> any:
        """Sanitize an individual value."""
        if value is None:
            return value
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            return "***"
        else:
            return cls.mask_string(str(value))

    @classmethod
    def sanitize_dict(cls, config_dict: dict) -> dict:
        """
        Recursively sanitize a dictionary, masking sensitive values.
        Returns a new dictionary; does not mutate the original.
        """
        sanitized = {}
        for k, v in config_dict.items():
            if isinstance(v, dict):
                sanitized[k] = cls.sanitize_dict(v)
            elif cls.is_sensitive(str(k)):
                sanitized[k] = cls.sanitize_value(v)
            else:
                sanitized[k] = v
        return sanitized


def get_config_dict(cls_obj) -> dict:
    """Extract configuration variables from a class object."""
    cfg = {}
    for key in dir(cls_obj):
        if key.startswith("_"):
            continue
        val = getattr(cls_obj, key)
        if callable(val) or isinstance(val, (classmethod, staticmethod, property)):
            continue
        cfg[key] = val
    return cfg


# ==================== Page Routes ====================
# Single authoritative definition of all Streamlit page paths.
# Import these constants directly from config rather than going through
# an intermediary module, so route updates propagate consistently from
# one place throughout the entire application.

PAGE_HOME = "pages/0_Home.py"
PAGE_LOGIN = "pages/0_Login.py"
PAGE_ANALYTICS_DASHBOARD = "pages/1_Analytics_Dashboard.py"
PAGE_MY_CASES = "pages/1_My_Cases.py"
PAGE_CASE_DETAILS = "pages/2_Case_Details.py"
PAGE_APPEAL_ESTIMATOR = "pages/2_Appeal_Estimator.py"
PAGE_REPORT_OUTCOME = "pages/3_Report_Outcome.py"
PAGE_SETTINGS = "pages/3_Settings.py"
PAGE_DEADLINE_TRACKER = "pages/3_Deadline_Tracker.py"
PAGE_CHAT = "pages/4_Chat.py"

# Print config to stdout/logger if debug mode is enabled, using the sanitizer

if Config.DEBUG:
    try:
        raw_config = get_config_dict(Config)
        safe_config = ConfigSanitizer.sanitize_dict(raw_config)
        logger.debug(f"Active Configuration (Sanitized): {safe_config}")
        print(f"DEBUG: LegalAssist AI Config Loaded: {safe_config}")
    except Exception as e:
        logger.error(f"Failed to dump sanitized configuration: {e}")

