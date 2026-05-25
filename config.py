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
    
    # --- Authentication (JWT & OTP) ---
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = _get_int_env("JWT_EXPIRY_HOURS", 7 * 24)
    OTP_EXPIRY_MINUTES = _get_int_env("OTP_EXPIRY_MINUTES", 10)
    OTP_MAX_ATTEMPTS = _get_int_env("OTP_MAX_ATTEMPTS", 3)
    
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

    @classmethod
    def get_twilio_auth_token(cls) -> str:
        """Return the Twilio auth token, retrieved on demand to limit exposure."""
        # Prefer centralized secret manager
        try:
            from utils.secret_manager import get_secret
            val = get_secret("twilio_auth_token") or _get_val("TWILIO_AUTH_TOKEN", "")
            return str(val or "")
        except Exception:
            return str(_get_val("TWILIO_AUTH_TOKEN", "") or "")

    # --- Notification Settings (Email) ---
    SENDGRID_FROM_EMAIL = _get_val("SENDGRID_FROM_EMAIL", "noreply@legalassist.ai")

    @classmethod
    def get_sendgrid_api_key(cls) -> str:
        """Return the SendGrid API key, retrieved on demand to limit exposure."""
        try:
            from utils.secret_manager import get_secret
            val = get_secret("sendgrid_api_key") or _get_val("SENDGRID_API_KEY", "")
            return str(val or "")
        except Exception:
            return str(_get_val("SENDGRID_API_KEY", "") or "")

    # --- Application URLs ---
    BASE_URL = _get_val("BASE_URL", "https://legalassist.ai")

    @classmethod
    def is_development(cls):
        return cls.APP_ENV in ("dev", "development", "local") or cls.DEBUG or cls.TESTING
