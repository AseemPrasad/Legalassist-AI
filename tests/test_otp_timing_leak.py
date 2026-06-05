"""Tests for the OTP timing side-channel fix (#1570).

Verifies that:
- All failure branches in verify_otp_and_create_token perform the hash
  comparison regardless of whether an OTP record exists.
- The "no OTP record" path always runs _verify_otp_hash() (constant work).
- All failure paths return the same generic error message.
- The source no longer contains an early return before hash comparison.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from unittest.mock import MagicMock, patch, call

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------

def test_no_early_return_before_hash_comparison():
    """The function must not return early before calling _verify_otp_hash.

    The old code had:
        if not otp_record:
            return False, ..., None   # before any hash work

    This is a timing leak: the "no record" path returned much faster than
    the "wrong OTP" path, letting attackers determine whether an email has
    a pending OTP.
    """
    import auth as auth_module

    source = inspect.getsource(auth_module.verify_otp_and_create_token)
    tree = ast.parse(source)

    # Find the position of _verify_otp_hash call vs the early-return pattern
    # Strategy: confirm the hash comparison call comes BEFORE any check of
    # "if not otp_record: return".
    call_positions = []
    early_return_positions = []

    for node in ast.walk(tree):
        # Look for calls to _verify_otp_hash
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "_verify_otp_hash":
                call_positions.append(node.lineno)

        # Look for "if not otp_record: return ..."
        if isinstance(node, ast.If):
            test = node.test
            # Check for `not otp_record`
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                operand = test.operand
                if isinstance(operand, ast.Name) and operand.id == "otp_record":
                    # Check the body contains a Return
                    for body_node in ast.walk(node):
                        if isinstance(body_node, ast.Return):
                            early_return_positions.append(node.lineno)

    assert call_positions, "_verify_otp_hash must be called in verify_otp_and_create_token"

    if early_return_positions:
        # If there is still an early return on "not otp_record", it must come
        # AFTER the hash comparison, not before.
        first_hash_call = min(call_positions)
        first_early_return = min(early_return_positions)
        assert first_hash_call < first_early_return, (
            "The hash comparison (_verify_otp_hash) must occur BEFORE any "
            "early return on 'if not otp_record', otherwise the no-record "
            "path is measurably faster than the wrong-OTP path."
        )


def test_dummy_hash_used_when_no_record():
    """When no OTP record exists, a dummy hash must be compared.

    This equalises the work done across the no-record and wrong-OTP paths.
    """
    import auth as auth_module

    source = inspect.getsource(auth_module.verify_otp_and_create_token)

    assert "dummy_hash" in source, (
        "verify_otp_and_create_token must compute a dummy_hash for use when "
        "no OTP record exists, so the hash comparison work is always performed."
    )


def test_hash_comparison_always_runs(monkeypatch):
    """_verify_otp_hash must be called even when get_pending_otp returns None."""
    import auth as auth_module

    hash_call_count = {"n": 0}
    original_verify = auth_module._verify_otp_hash

    def _counting_verify(otp, hash_val):
        hash_call_count["n"] += 1
        return original_verify(otp, hash_val)

    monkeypatch.setattr(auth_module, "_verify_otp_hash", _counting_verify)
    monkeypatch.setattr(auth_module, "get_pending_otp", lambda db, email: None)
    monkeypatch.setattr(auth_module, "SessionLocal", lambda: MagicMock())

    auth_module.verify_otp_and_create_token("nobody@example.com", "123456")

    assert hash_call_count["n"] >= 1, (
        "_verify_otp_hash must be called even when no OTP record exists "
        "so the no-record path takes the same time as the wrong-OTP path."
    )


# ---------------------------------------------------------------------------
# Generic error message across all failure paths
# ---------------------------------------------------------------------------

def test_all_failure_paths_return_same_message(monkeypatch):
    """All OTP failure branches must return the same generic error string."""
    import auth as auth_module

    GENERIC = "Invalid or expired OTP. Please request a new one."

    # Path 1: no OTP record
    monkeypatch.setattr(auth_module, "get_pending_otp", lambda db, email: None)
    monkeypatch.setattr(auth_module, "SessionLocal", lambda: MagicMock())

    _, msg1, _ = auth_module.verify_otp_and_create_token("a@b.com", "000000")
    assert msg1 == GENERIC, f"No-record path returned: {msg1!r}"

    # Path 2: locked OTP record
    from datetime import datetime, timezone, timedelta
    locked_record = MagicMock()
    locked_record.is_locked.return_value = True
    locked_record.locked_until = datetime.now(timezone.utc) + timedelta(minutes=10)
    locked_record.otp_hash = auth_module._hash_otp("999999")

    monkeypatch.setattr(auth_module, "get_pending_otp", lambda db, email: locked_record)

    _, msg2, _ = auth_module.verify_otp_and_create_token("a@b.com", "000000")
    assert msg2 == GENERIC, f"Locked path returned: {msg2!r}"

    # Path 3: wrong OTP
    wrong_record = MagicMock()
    wrong_record.is_locked.return_value = False
    wrong_record.otp_hash = auth_module._hash_otp("999999")
    wrong_record.id = 1
    wrong_record.failed_attempts = 0

    monkeypatch.setattr(auth_module, "get_pending_otp", lambda db, email: wrong_record)
    monkeypatch.setattr(auth_module, "record_otp_failed_attempt", lambda *a, **kw: None)

    _, msg3, _ = auth_module.verify_otp_and_create_token("a@b.com", "000000")
    assert msg3 == GENERIC, f"Wrong-OTP path returned: {msg3!r}"
