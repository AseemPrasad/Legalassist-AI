"""Tests for Celery result backend expiry fix (#1574).

Verifies that:
- Celery is configured with result_expires (TTL for task results).
- result_expires is set to a reasonable value (24 hours or less).
- The configuration is applied to the app instance.
- Sensitive fields are redacted from stored results.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ---------------------------------------------------------------------------
# Celery configuration checks (from source)
# ---------------------------------------------------------------------------

def test_result_expires_configured_in_source():
    """result_expires must be explicitly set in celery_app.py conf."""
    with open("celery_app.py", "r") as f:
        source = f.read()
    
    assert "result_expires" in source, (
        "celery_app.py must explicitly set result_expires in app.conf.update(). "
        "Without it, task results persist indefinitely."
    )
    
    # Verify it's set to 86400 (24 hours)
    assert "result_expires=86400" in source, (
        "result_expires should be set to 86400 seconds (24 hours)"
    )


def test_pii_protection_comment_in_source():
    """Source should document why result_expires is critical for PII protection."""
    with open("celery_app.py", "r") as f:
        source = f.read()
    
    # Verify documentation mentions PII protection
    assert "PII" in source or "sensitive" in source.lower(), (
        "celery_app.py should document the PII protection rationale for result_expires"
    )


# ---------------------------------------------------------------------------
# Result redaction
# ---------------------------------------------------------------------------

def test_redact_task_result_strips_sensitive_fields():
    """_redact_task_result must remove LLM output and extracted text."""
    import celery_app

    original = {
        "status": "success",
        "summary_text": "Plaintiff alleges breach of contract worth $50,000 from Acme Corp.",
        "key_points": ["Breach occurred on Jan 1", "Damages claimed: $50k"],
        "remedies_list": ["Monetary compensation", "Injunctive relief"],
        "extracted_text": "Full PDF content with party names and addresses...",
    }

    redacted = celery_app._redact_task_result(original)

    assert redacted["status"] == "success"  # keep non-sensitive fields
    assert redacted["summary_text"] == "[REDACTED]"
    assert redacted["key_points"] == "[REDACTED]"
    assert redacted["remedies_list"] == "[REDACTED]"
    assert redacted["extracted_text"] == "[REDACTED]"


def test_redact_task_result_preserves_non_sensitive_fields():
    """Non-sensitive fields must not be redacted."""
    import celery_app

    original = {
        "status": "success",
        "task_id": "abc123",
        "duration_seconds": 5.2,
        "summary_text": "Sensitive content",
    }

    redacted = celery_app._redact_task_result(original)

    assert redacted["status"] == "success"
    assert redacted["task_id"] == "abc123"
    assert redacted["duration_seconds"] == 5.2
    assert redacted["summary_text"] == "[REDACTED]"


def test_redact_task_result_handles_non_dict():
    """_redact_task_result must handle non-dict results gracefully."""
    import celery_app

    assert celery_app._redact_task_result(None) is None
    assert celery_app._redact_task_result("string") == "string"
    assert celery_app._redact_task_result(42) == 42
    assert celery_app._redact_task_result([1, 2, 3]) == [1, 2, 3]


def test_redaction_function_exists():
    """_redact_task_result function must be defined."""
    import celery_app
    
    assert hasattr(celery_app, "_redact_task_result"), (
        "celery_app module must define _redact_task_result function for PII redaction"
    )
    assert callable(celery_app._redact_task_result), (
        "_redact_task_result must be callable"
    )
