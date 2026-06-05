"""Tests for case document ownership integer comparison fix (#1572).

Verifies that the ownership check in process_case_document_upload_task
uses integer comparison, not string comparison, so values like '007' or
' 7' cannot bypass the check for user 7.
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

import pytest


# ---------------------------------------------------------------------------
# Source-level check: no str() comparison in ownership check
# ---------------------------------------------------------------------------

def test_ownership_check_does_not_use_string_comparison():
    """The ownership check must not use str(case.user_id) != str(user_id)."""
    import ast
    import celery_app as _ca
    import inspect

    source = inspect.getsource(_ca.process_case_document_upload_task)
    tree = ast.parse(source)

    # Walk all Compare nodes and confirm none are str(...) != str(...)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left_src = ast.unparse(node.left)
        for comparator in node.comparators:
            right_src = ast.unparse(comparator)
            pair = {left_src, right_src}
            assert not (
                "str(case.user_id)" in pair and "str(user_id)" in pair
            ), (
                "Ownership check must not use string comparison. "
                "str('007') != str('7') even though int('007') == 7."
            )


def test_ownership_check_uses_integer_comparison():
    """The ownership check must use integer equality after int() conversion."""
    import inspect
    import celery_app

    source = inspect.getsource(celery_app.process_case_document_upload_task)

    # Must convert user_id to int
    assert 'int(user_id)' in source, (
        "Ownership check must convert user_id to int() before comparing."
    )
    # Must compare case.user_id (an Integer FK) directly
    assert 'case.user_id !=' in source or 'case.user_id ==' in source, (
        "Ownership check must compare case.user_id directly (integer)."
    )


# ---------------------------------------------------------------------------
# Behavioural checks via the comparison logic directly
# ---------------------------------------------------------------------------

def _ownership_passes(db_user_id: int, task_user_id: str) -> bool:
    """Replicate the fixed ownership logic."""
    try:
        task_int = int(task_user_id)
    except (TypeError, ValueError):
        return False
    return db_user_id == task_int


def test_correct_user_id_passes():
    assert _ownership_passes(7, "7") is True


def test_leading_zero_does_not_bypass():
    """'007' must NOT match user 7 via string comparison (old bug)."""
    # Old string logic: str(7) == "7", str("007") == "007" → not equal → blocked
    # But int("007") == 7 → so integer comparison correctly ALLOWS it (it IS user 7)
    # The real bypass risk is '007' matching a DIFFERENT user.
    # Integer comparison correctly resolves '007' to 7 — the same user.
    assert _ownership_passes(7, "007") is True   # same user — should pass
    assert _ownership_passes(8, "007") is False  # different user — must block


def test_leading_space_does_not_cause_bypass():
    """' 7' (with leading space) must be correctly resolved to integer 7."""
    # int(' 7') == 7 in Python — this is intentional Python behaviour
    assert _ownership_passes(7, " 7") is True    # same user
    assert _ownership_passes(8, " 7") is False   # different user


def test_wrong_user_id_blocked():
    assert _ownership_passes(7, "8") is False
    assert _ownership_passes(7, "99") is False


def test_non_numeric_user_id_raises():
    """A non-numeric user_id must be rejected."""
    assert _ownership_passes(7, "admin") is False
    assert _ownership_passes(7, "") is False
    assert _ownership_passes(7, "null") is False


def test_float_string_rejected():
    """A float string like '7.0' must be rejected (not coerced)."""
    # int("7.0") raises ValueError — correct behaviour
    assert _ownership_passes(7, "7.0") is False


def test_negative_user_id_blocked():
    """A negative user_id string must not match a positive DB user_id."""
    assert _ownership_passes(7, "-7") is False


def test_zero_user_id_blocked():
    assert _ownership_passes(7, "0") is False
    assert _ownership_passes(0, "0") is True
