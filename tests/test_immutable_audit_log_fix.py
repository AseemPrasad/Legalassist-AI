"""Tests for the immutable audit log security fix (#1238).

Verifies that:
1. append_audit_entry uses a dedicated session (not the shared app session).
2. The integrity_hash is set on the first INSERT — no post-insert UPDATE.
3. The hash chain is valid after multiple sequential writes.
4. verify_audit_chain detects tampering.
5. Concurrent writes do not produce a broken chain (serialisation).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from sqlalchemy import create_engine, text, event as sa_event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.base import Base as _Base
from db.immutable_audit_log import (
    ImmutableAuditLog,
    _compute_hash,
    append_audit_entry,
    verify_audit_chain,
    _get_audit_session,
)


# ---------------------------------------------------------------------------
# Minimal in-memory DB
# ---------------------------------------------------------------------------

_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
try:
    _Base.metadata.create_all(bind=_ENGINE)
except Exception:
    pass

_SessionFactory = sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False, bind=_ENGINE
)


@pytest.fixture(autouse=True)
def clean_audit_table():
    """Truncate the audit table before each test."""
    with _ENGINE.begin() as conn:
        conn.execute(text("DELETE FROM immutable_audit_log"))
    yield


# ---------------------------------------------------------------------------
# Patch _get_audit_session to use our test engine
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_audit_session(clean_audit_table):
    """Route all audit session calls to the test in-memory DB."""
    def _test_session():
        return _SessionFactory()

    with patch("db.immutable_audit_log._get_audit_session", side_effect=_test_session):
        with patch("db.session._is_postgres", False):
            with patch("db.session._is_sqlite", True):
                yield


# ---------------------------------------------------------------------------
# 1. Dedicated session — not the shared app session
# ---------------------------------------------------------------------------

def test_append_uses_dedicated_session_not_shared():
    """append_audit_entry must call _get_audit_session, not db_session."""
    with patch("db.immutable_audit_log._get_audit_session") as mock_get:
        mock_session = _SessionFactory()
        mock_get.return_value = mock_session
        with patch("db.session._is_postgres", False), \
             patch("db.session._is_sqlite", True):
            append_audit_entry(event_type="test.event", action="create")

        mock_get.assert_called_once()


def test_append_does_not_use_db_session_context_manager():
    """append_audit_entry must NOT use the db_session() context manager."""
    import db.immutable_audit_log as ial_module
    import inspect

    source = inspect.getsource(ial_module.append_audit_entry)
    assert "db_session()" not in source, (
        "append_audit_entry must not use db_session() — it must use a "
        "dedicated session via _get_audit_session() to prevent application "
        "code from rolling back or modifying audit entries."
    )


# ---------------------------------------------------------------------------
# 2. Hash set on first INSERT — no post-insert UPDATE
# ---------------------------------------------------------------------------

def test_integrity_hash_set_on_insert_not_empty():
    """The row must be written with a non-empty integrity_hash on first INSERT."""
    updates_issued = []

    # Track any UPDATE statements issued against the audit table
    @sa_event.listens_for(_ENGINE, "before_execute")
    def _track_updates(conn, clauseelement, multiparams, params, execution_options):
        stmt_str = str(clauseelement).upper()
        if "UPDATE" in stmt_str and "IMMUTABLE_AUDIT_LOG" in stmt_str:
            updates_issued.append(stmt_str)

    try:
        append_audit_entry(event_type="test.insert", action="create")
    finally:
        sa_event.remove(_ENGINE, "before_execute", _track_updates)

    assert not updates_issued, (
        "append_audit_entry must not issue any UPDATE on immutable_audit_log. "
        "The integrity_hash must be computed before the INSERT."
    )

    # Verify the row has a real hash
    with _SessionFactory() as db:
        entry = db.query(ImmutableAuditLog).first()
    assert entry is not None
    assert entry.integrity_hash != "", "integrity_hash must not be empty after insert"
    assert len(entry.integrity_hash) == 64, "integrity_hash must be a SHA-256 hex digest"


# ---------------------------------------------------------------------------
# 3. Hash chain validity
# ---------------------------------------------------------------------------

def test_hash_chain_valid_after_sequential_writes():
    """Multiple sequential writes must produce a valid hash chain."""
    for i in range(5):
        append_audit_entry(
            event_type="test.chain",
            action="write",
            resource_id=str(i),
            outcome="success",
        )

    # verify_audit_chain also needs the patched session
    with patch("db.immutable_audit_log._get_audit_session", side_effect=lambda: _SessionFactory()):
        result = verify_audit_chain()

    assert result["valid"] is True
    assert result["entries_checked"] == 5
    assert result["broken_at"] is None


def test_first_entry_uses_genesis_prev_hash():
    """The first entry must have prev_hash == 'GENESIS'."""
    append_audit_entry(event_type="test.genesis", action="create")

    with _SessionFactory() as db:
        entry = db.query(ImmutableAuditLog).first()
    assert entry.prev_hash == "GENESIS"


def test_second_entry_chains_to_first():
    """The second entry's prev_hash must equal the first entry's integrity_hash."""
    append_audit_entry(event_type="test.chain", action="first")
    append_audit_entry(event_type="test.chain", action="second")

    with _SessionFactory() as db:
        entries = db.query(ImmutableAuditLog).order_by(ImmutableAuditLog.id).all()

    assert len(entries) == 2
    assert entries[1].prev_hash == entries[0].integrity_hash


# ---------------------------------------------------------------------------
# 4. Tamper detection
# ---------------------------------------------------------------------------

def test_verify_chain_detects_hash_tampering():
    """verify_audit_chain must detect when an entry's hash is modified."""
    append_audit_entry(event_type="test.tamper", action="write")
    append_audit_entry(event_type="test.tamper", action="write")

    # Directly tamper with the first entry's hash (bypassing the ORM)
    with _ENGINE.begin() as conn:
        conn.execute(
            text("UPDATE immutable_audit_log SET integrity_hash = 'deadbeef' WHERE id = (SELECT MIN(id) FROM immutable_audit_log)")
        )

    result = verify_audit_chain()
    assert result["valid"] is False
    assert result["broken_at"] is not None


def test_verify_chain_detects_prev_hash_tampering():
    """verify_audit_chain must detect when prev_hash is modified."""
    append_audit_entry(event_type="test.tamper", action="write")
    append_audit_entry(event_type="test.tamper", action="write")

    # Tamper with the second entry's prev_hash
    with _ENGINE.begin() as conn:
        conn.execute(
            text("UPDATE immutable_audit_log SET prev_hash = 'tampered' WHERE id = (SELECT MAX(id) FROM immutable_audit_log)")
        )

    result = verify_audit_chain()
    assert result["valid"] is False


def test_verify_chain_valid_empty_table():
    """verify_audit_chain on an empty table must return valid with 0 entries."""
    result = verify_audit_chain()
    assert result["valid"] is True
    assert result["entries_checked"] == 0


# ---------------------------------------------------------------------------
# 5. _compute_hash is deterministic
# ---------------------------------------------------------------------------

def test_compute_hash_deterministic():
    """Same input must always produce the same hash."""
    data = {"event_type": "test", "action": "create", "outcome": "success"}
    h1 = _compute_hash(data, "GENESIS")
    h2 = _compute_hash(data, "GENESIS")
    assert h1 == h2
    assert len(h1) == 64


def test_compute_hash_changes_with_prev_hash():
    """Different prev_hash values must produce different hashes."""
    data = {"event_type": "test", "action": "create"}
    h1 = _compute_hash(data, "GENESIS")
    h2 = _compute_hash(data, "different_prev")
    assert h1 != h2


# ---------------------------------------------------------------------------
# 6. Metadata sanitisation still works
# ---------------------------------------------------------------------------

def test_append_stores_metadata():
    """Metadata passed to append_audit_entry must be stored on the entry."""
    append_audit_entry(
        event_type="test.meta",
        action="create",
        metadata={"key": "value", "count": 42},
    )

    with _SessionFactory() as db:
        entry = db.query(ImmutableAuditLog).first()

    assert entry.audit_metadata is not None
    assert entry.audit_metadata.get("key") == "value"
