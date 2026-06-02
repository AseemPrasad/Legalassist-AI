"""
Tests for JWT config parity between Streamlit (config.py/auth.py) and
the FastAPI backend (api/config.py/api/auth.py).

Ensures both stacks honour the canonical JWT_SECRET / JWT_SECRET_PREVIOUS
env vars and fall back correctly to the legacy aliases.
"""
import os
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

import importlib

from api.config import APISettings
from config import Config


def test_api_settings_canonical_precedence(monkeypatch):
    """JWT_SECRET takes precedence over JWT_SECRET_KEY in APISettings."""
    monkeypatch.setenv("JWT_SECRET", "canonical-secret-1234567890123456")
    monkeypatch.setenv("JWT_SECRET_KEY", "legacy-secret-1234567890123456")
    monkeypatch.setenv("JWT_SECRET_PREVIOUS", "canonical-prev-1234567890123456")
    monkeypatch.setenv("JWT_SECRET_KEY_PREVIOUS", "legacy-prev-1234567890123456")

    settings = APISettings()
    assert settings.JWT_SECRET_KEY == "canonical-secret-1234567890123456"
    assert settings.JWT_SECRET_KEY_PREVIOUS == "canonical-prev-1234567890123456"


def test_api_settings_legacy_fallback(monkeypatch):
    """JWT_SECRET_KEY is used when canonical JWT_SECRET is absent."""
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("JWT_SECRET_KEY", "legacy-secret-1234567890123456")
    monkeypatch.delenv("JWT_SECRET_PREVIOUS", raising=False)
    monkeypatch.setenv("JWT_SECRET_KEY_PREVIOUS", "legacy-prev-1234567890123456")

    settings = APISettings()
    assert settings.JWT_SECRET_KEY == "legacy-secret-1234567890123456"
    assert settings.JWT_SECRET_KEY_PREVIOUS == "legacy-prev-1234567890123456"


def test_streamlit_config_canonical_precedence(monkeypatch):
    """get_current_jwt_secret() returns JWT_SECRET over JWT_SECRET_KEY."""
    monkeypatch.setenv("JWT_SECRET", "canonical-secret-1234567890123456")
    monkeypatch.setenv("JWT_SECRET_KEY", "legacy-secret-1234567890123456")

    assert Config.get_current_jwt_secret() == "canonical-secret-1234567890123456"


def test_streamlit_config_legacy_fallback(monkeypatch):
    """get_current_jwt_secret() falls back to JWT_SECRET_KEY."""
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("JWT_SECRET_KEY", "legacy-secret-1234567890123456")

    assert Config.get_current_jwt_secret() == "legacy-secret-1234567890123456"


def test_streamlit_config_previous_secret_alias(monkeypatch):
    """JWT_SECRET_PREVIOUS falls back to JWT_SECRET_KEY_PREVIOUS (class-level reload)."""
    monkeypatch.delenv("JWT_SECRET_PREVIOUS", raising=False)
    monkeypatch.setenv("JWT_SECRET_KEY_PREVIOUS", "legacy-prev-1234567890123456")

    import config as config_mod
    importlib.reload(config_mod)
    assert config_mod.Config.JWT_SECRET_PREVIOUS == "legacy-prev-1234567890123456"


def test_cross_stack_shared_secret(monkeypatch):
    """A token signed with the canonical JWT_SECRET is decodable by both stacks."""
    import jwt as pyjwt

    secret = "shared-secret-key-1234567890123456"
    monkeypatch.setenv("JWT_SECRET", secret)

    # FastAPI side resolves the secret via APISettings
    api_settings = APISettings()
    assert api_settings.JWT_SECRET_KEY == secret

    # Streamlit side resolves the secret dynamically
    assert Config.get_current_jwt_secret() == secret

    # A token encoded with that secret must be decodable by both
    token = pyjwt.encode({"sub": "42", "email": "tester@example.com"}, secret, algorithm="HS256")
    decoded_api = pyjwt.decode(token, api_settings.JWT_SECRET_KEY, algorithms=["HS256"], options={"verify_exp": False})
    decoded_streamlit = pyjwt.decode(token, Config.get_current_jwt_secret(), algorithms=["HS256"], options={"verify_exp": False})

    assert decoded_api["sub"] == "42"
    assert decoded_streamlit["sub"] == "42"
