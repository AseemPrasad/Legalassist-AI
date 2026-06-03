"""Tests for the JWT secret file-fallback removal fix (#1240).

Verifies that:
- The .jwt_secret file is no longer used as a fallback secret source.
- CASE_ANONYMIZATION_SECRET env var is the only accepted source.
- Weak / low-entropy secrets are rejected.
- The .jwt_secret path is listed in .gitignore.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

# Stub optional heavy deps
for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
import services.case_anonymization as ca


# ---------------------------------------------------------------------------
# File fallback is gone
# ---------------------------------------------------------------------------

def test_jwt_secret_file_not_used_as_fallback(tmp_path, monkeypatch):
    """.jwt_secret file must NOT be read even when it exists and env var is unset."""
    # Ensure env var is absent
    monkeypatch.delenv("CASE_ANONYMIZATION_SECRET", raising=False)

    # Create a valid .jwt_secret file in a temp dir
    secret_file = tmp_path / ".jwt_secret"
    secret_file.write_text("a" * 40, encoding="utf-8")

    # Patch _get_val so Streamlit path also returns nothing
    with patch("services.case_anonymization._get_val" if hasattr(ca, "_get_val") else "config._get_val",
               return_value="", create=True):
        with pytest.raises(RuntimeError, match="CASE_ANONYMIZATION_SECRET is not configured"):
            ca._get_case_anonymization_secret()


def test_env_var_is_accepted(monkeypatch):
    """CASE_ANONYMIZATION_SECRET env var must be accepted."""
    import secrets as _secrets
    strong = _secrets.token_hex(32)
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", strong)
    secret = ca._get_case_anonymization_secret()
    assert secret == strong


def test_missing_env_var_raises(monkeypatch):
    """Missing env var must raise RuntimeError with a clear message."""
    monkeypatch.delenv("CASE_ANONYMIZATION_SECRET", raising=False)
    with patch("config._get_val", return_value=""):
        with pytest.raises(RuntimeError, match="CASE_ANONYMIZATION_SECRET is not configured"):
            ca._get_case_anonymization_secret()


# ---------------------------------------------------------------------------
# Entropy check
# ---------------------------------------------------------------------------

def test_low_entropy_secret_rejected(monkeypatch):
    """A secret made of one repeated character must be rejected."""
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", "a" * 40)
    with pytest.raises(ValueError, match="insufficient entropy"):
        ca._get_case_anonymization_secret()


def test_high_entropy_secret_accepted(monkeypatch):
    """A randomly generated secret with sufficient entropy must be accepted."""
    import secrets as _secrets
    strong = _secrets.token_hex(32)  # 64 hex chars, high entropy
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", strong)
    result = ca._get_case_anonymization_secret()
    assert result == strong


def test_secret_too_short_rejected(monkeypatch):
    """A secret shorter than _MIN_SECRET_LENGTH must be rejected."""
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", "short")
    with pytest.raises(ValueError, match="at least"):
        ca._get_case_anonymization_secret()


# ---------------------------------------------------------------------------
# Test-mode override still works
# ---------------------------------------------------------------------------

def test_override_allowed_in_testing(monkeypatch):
    """Secret override must work when Config.TESTING is True."""
    from config import Config
    prev = Config.TESTING
    try:
        Config.TESTING = True
        result = ca._get_case_anonymization_secret(override="t" * 40)
        # "t" * 40 has only 1 distinct char — but override skips entropy check
        assert result == "t" * 40
    finally:
        Config.TESTING = prev


def test_override_blocked_outside_testing(monkeypatch):
    """Secret override must be blocked when Config.TESTING is False."""
    from config import Config
    prev = Config.TESTING
    try:
        Config.TESTING = False
        with pytest.raises(RuntimeError, match="testing mode"):
            ca._get_case_anonymization_secret(override="t" * 40)
    finally:
        Config.TESTING = prev


# ---------------------------------------------------------------------------
# .gitignore contains .jwt_secret
# ---------------------------------------------------------------------------

def test_jwt_secret_in_gitignore():
    """.jwt_secret must be listed in .gitignore to prevent accidental commits."""
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    assert gitignore.exists(), ".gitignore not found at project root"
    content = gitignore.read_text(encoding="utf-8")
    assert ".jwt_secret" in content, (
        ".jwt_secret is not listed in .gitignore — it could be accidentally committed "
        "and expose secrets to anyone with repo read access."
    )


# ---------------------------------------------------------------------------
# Existing anonymization tests still pass (smoke check)
# ---------------------------------------------------------------------------

def test_generate_anonymized_case_id_deterministic(monkeypatch):
    """ID generation must remain deterministic with the same env secret."""
    import secrets as _secrets
    strong = _secrets.token_hex(32)
    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", strong)
    from datetime import datetime, timezone
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    id1 = ca._generate_anonymized_case_id(case_id=1, created_at=dt)
    id2 = ca._generate_anonymized_case_id(case_id=1, created_at=dt)
    assert id1 == id2
    assert len(id1) == 12


def test_generate_anonymized_case_id_changes_with_secret(monkeypatch):
    """Different secrets must produce different IDs."""
    import secrets as _secrets
    from datetime import datetime, timezone
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", _secrets.token_hex(32))
    id_a = ca._generate_anonymized_case_id(case_id=1, created_at=dt)

    monkeypatch.setenv("CASE_ANONYMIZATION_SECRET", _secrets.token_hex(32))
    id_b = ca._generate_anonymized_case_id(case_id=1, created_at=dt)

    assert id_a != id_b
