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

# Detection of the environment should be done only once at startup.
try:
    import streamlit as st
    # Verify st.secrets is accessible
    _ = st.secrets
    _HAS_STREAMLIT = True
except (ImportError, RuntimeError, AttributeError, FileNotFoundError):
    st = None
    _HAS_STREAMLIT = False

def _get_val(key, default=None):
    """
    Retrieve configuration value from Streamlit secrets or environment variables.
    Refactored to avoid redundant dynamic imports.
    """
    if _HAS_STREAMLIT and st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
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
    REQUIRE_HTTPS = _get_bool_env("REQUIRE_HTTPS", True)
    
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
    
    # API Key for OpenAI.
    OPENAI_API_KEY = _get_val("OPENAI_API_KEY", "")
    
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
    MAX_DOCUMENT_TEXT_STORAGE_LIMIT = _get_int_env("MAX_DOCUMENT_TEXT_STORAGE_LIMIT", 50000)
    # --- Attachments ---
    # Directory where uploaded attachments are stored (development)
    ATTACHMENTS_DIR = _get_val("ATTACHMENTS_DIR", str(PROJECT_ROOT / "attachments"))
    # Use randomized filenames to avoid collisions and leaking original names
    ATTACHMENTS_RANDOMIZE_FILENAMES = _get_bool_env("ATTACHMENTS_RANDOMIZE_FILENAMES", True)
    
    # --- Export Settings ---
    # Directory where user data exports are saved (local storage)
    EXPORTS_DIR = _get_val("EXPORTS_DIR", str(PROJECT_ROOT / ".exports"))
    # Hours before export files expire and can be deleted
    EXPORT_FILE_EXPIRY_HOURS = _get_int_env("EXPORT_FILE_EXPIRY_HOURS", 24)
    # Default privacy redaction profile for anonymized exports and reports
    DEFAULT_PRIVACY_PROFILE = _get_val("DEFAULT_PRIVACY_PROFILE", "personal_identifiers")
    # Optional JSON override for privacy profile definitions
    PRIVACY_REDACTION_PROFILES_JSON = _get_val("PRIVACY_REDACTION_PROFILES_JSON", "")
    
    # --- Database Settings ---
    DATABASE_URL = _get_val("DATABASE_URL", "sqlite:///./legalassist.db")

    # --- Backend API Settings ---
    API_BASE_URL = _get_val("API_BASE_URL", "")
    API_REQUEST_TIMEOUT_SECONDS = float(_get_val("API_REQUEST_TIMEOUT_SECONDS", "5.0"))
    
    # --- Authentication (JWT & OTP) ---
    JWT_ALGORITHM = "HS256"
    JWT_ISSUER = _get_val("JWT_ISSUER", "legalassist.ai")
    JWT_AUDIENCE = _get_val("JWT_AUDIENCE", "legalassist-users")
    JWT_EXPIRY_HOURS = _get_int_env("JWT_EXPIRY_HOURS", 7 * 24)
    JWT_SECRET_PREVIOUS = _get_val("JWT_SECRET_PREVIOUS", _get_val("JWT_SECRET_KEY_PREVIOUS", _get_val("JWT_SECRET_OLD", "")))
    OTP_EXPIRY_MINUTES = _get_int_env("OTP_EXPIRY_MINUTES", 10)
    OTP_MAX_ATTEMPTS = _get_int_env("OTP_MAX_ATTEMPTS", 3)
    OTP_REQUEST_RATE_LIMIT_MAX = _get_int_env("OTP_REQUEST_RATE_LIMIT_MAX", 5)
    OTP_REQUEST_RATE_LIMIT_HOURS = _get_int_env("OTP_REQUEST_RATE_LIMIT_HOURS", 1)
    
    @classmethod
    def get_jwt_secret(cls):
        """
        Resolve JWT secret securely.
        
        JWT_SECRET must be provided via environment variable or Streamlit secrets.
        File-based secrets are no longer supported for security.
        
        Raises:
            RuntimeError: If JWT_SECRET is not configured in environment variables.
        """
        secret = cls.get_current_jwt_secret()
        if secret:
            return secret
        
        env_name = cls.APP_ENV.upper()
        raise RuntimeError(
            f"JWT_SECRET is not configured for the {env_name} environment. "
            "For security, secrets must be explicitly provided via the 'JWT_SECRET' "
            "environment variable. Consider using AWS Secrets Manager or HashiCorp Vault "
            "for production secret management."
        )

    @classmethod
    def get_current_jwt_secret(cls) -> str:
        """Return the active JWT signing secret without falling back to placeholders."""
        return str(_get_val("JWT_SECRET", _get_val("JWT_SECRET_KEY", _get_val("JWT_SECRET_CURRENT", "")))).strip()

    @classmethod
    def get_jwt_secrets(cls) -> list[str]:
        """Return JWT secrets in verification order: current first, then previous."""
        secrets_to_try = [
            cls.get_current_jwt_secret(),
            str(cls.JWT_SECRET_PREVIOUS).strip(),
        ]
        return [secret for secret in dict.fromkeys(secrets_to_try) if secret]

    # --- Notification Settings (SMS) ---
    TWILIO_ACCOUNT_SID = _get_val("TWILIO_ACCOUNT_SID", "")
    TWILIO_FROM_NUMBER = _get_val("TWILIO_FROM_NUMBER", "")

    @classmethod
    def get_twilio_auth_token(cls) -> str:
        """Return the Twilio auth token, retrieved on demand to limit exposure."""
        return str(_get_val("TWILIO_AUTH_TOKEN", "") or "")

    # --- Notification Settings (Email) ---
    SENDGRID_FROM_EMAIL = _get_val("SENDGRID_FROM_EMAIL", "noreply@legalassist.ai")

    @classmethod
    def get_sendgrid_api_key(cls) -> str:
        """Return the SendGrid API key, retrieved on demand to limit exposure."""
        return str(_get_val("SENDGRID_API_KEY", "") or "")

    @classmethod
    def validate_runtime_security(cls):
        """Fail fast when production settings are insecure or missing required secrets."""
        if cls.is_production() and (cls.DEBUG or cls.TESTING):
            raise RuntimeError("DEBUG and TESTING must be disabled in production")

        if cls.is_production():
            required = {
                "JWT_SECRET": cls.get_current_jwt_secret(),
                "OPENROUTER_API_KEY": str(cls.OPENROUTER_API_KEY).strip(),
                "SENDGRID_API_KEY": cls.get_sendgrid_api_key(),
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise RuntimeError(
                    "Missing required production secrets: " + ", ".join(sorted(missing))
                )

            if cls.REQUIRE_HTTPS:
                if not str(cls.BASE_URL).lower().startswith("https://"):
                    raise RuntimeError("BASE_URL must use https:// in production")
                if cls.API_BASE_URL and not str(cls.API_BASE_URL).lower().startswith("https://"):
                    raise RuntimeError("API_BASE_URL must use https:// in production")

    # --- Application URLs ---
    BASE_URL = _get_val("BASE_URL", "https://legalassist.ai")

    @classmethod
    def is_development(cls):
        return cls.APP_ENV in ("dev", "development", "local") or cls.DEBUG or cls.TESTING

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

# Print config to stdout/logger if debug mode is enabled, using the sanitizer
if Config.DEBUG:
    try:
        raw_config = get_config_dict(Config)
        safe_config = ConfigSanitizer.sanitize_dict(raw_config)
        logger.debug(f"Active Configuration (Sanitized): {safe_config}")
        print(f"DEBUG: LegalAssist AI Config Loaded: {safe_config}")
    except Exception as e:
        logger.error(f"Failed to dump sanitized configuration: {e}")

