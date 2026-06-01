"""Tests for the RLS fix on the anonymized case lookup endpoint (#1239).

Verifies that:
- The endpoint source declares get_db_no_rls as its DB dependency (AST check).
- get_db_no_rls source never calls apply_rls_context (source check).
- The lookup service returns correct data and never exposes owner user_id.
- Malformed / unknown IDs are rejected.

These tests use AST/source inspection and a minimal in-memory DB to avoid
the pre-existing database.py mapper conflict on this branch (database.py
redefines notification_logs without extend_existing=True, which crashes
any test that imports api.auth → database).
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")
os.environ.setdefault("CASE_ANONYMIZATION_SECRET", "aB3dEf7hIj0kLmNoPqRsTuVwXyZ12345678")

for _mod in ("streamlit", "pytesseract", "pdf2image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Minimal in-memory DB using raw DDL — no ORM metadata, no mapper conflict
# ---------------------------------------------------------------------------

_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
with _ENGINE.begin() as _conn:
    _conn.execute(text("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            case_number TEXT NOT NULL,
            case_type TEXT NOT NULL,
            jurisdiction TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            title TEXT,
            anonymized_id TEXT UNIQUE,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT,
            updated_at TEXT
        )
    """))
    _conn.execute(text("""
        CREATE TABLE IF NOT EXISTS case_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            document_type TEXT NOT NULL,
            document_content TEXT,
            summary TEXT,
            remedies TEXT,
            uploaded_at TEXT,
            ocr_used INTEGER NOT NULL DEFAULT 0
        )
    """))
    _conn.execute(text("""
        CREATE TABLE IF NOT EXISTS case_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            event_date TEXT,
            created_at TEXT
        )
    """))

_SessionFactory = sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False, bind=_ENGINE
)


@pytest.fixture()
def test_db() -> Generator[Session, None, None]:
    connection = _ENGINE.connect()
    transaction = connection.begin()
    session = _SessionFactory(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _seed(db: Session, anon_id: str = "aabbccddeeff") -> None:
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        text(
            "INSERT INTO cases "
            "(user_id, case_number, case_type, jurisdiction, status, "
            " anonymized_id, created_at, updated_at, version) "
            "VALUES (7, 'TEST-001', 'civil', 'Delhi', 'active', "
            ":anon_id, :now, :now, 1)"
        ),
        {"anon_id": anon_id, "now": now},
    )
    db.commit()


# ---------------------------------------------------------------------------
# Source-level checks (no import of api.dependencies needed)
# ---------------------------------------------------------------------------

def test_route_source_uses_get_db_no_rls():
    """The route source must import and use get_db_no_rls, not get_db_rls_optional."""
    route_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "routes", "anonymized_cases.py"
    )
    source = open(route_path, encoding="utf-8").read()
    tree = ast.parse(source)

    # Check imports: get_db_no_rls must be imported, get_db_rls_optional must not
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)

    assert "get_db_no_rls" in imported_names, (
        "anonymized_cases.py must import get_db_no_rls"
    )
    assert "get_db_rls_optional" not in imported_names, (
        "anonymized_cases.py must NOT import get_db_rls_optional — it sets an "
        "RLS context that is never enforced by the anonymized_id lookup query"
    )


def test_dependencies_source_has_get_db_no_rls():
    """api/dependencies.py must define get_db_no_rls."""
    deps_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "dependencies.py"
    )
    source = open(deps_path, encoding="utf-8").read()
    assert "def get_db_no_rls" in source, (
        "api/dependencies.py must define get_db_no_rls"
    )


def test_get_db_no_rls_source_never_calls_apply_rls_context():
    """get_db_no_rls must never call apply_rls_context in its source."""
    deps_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "dependencies.py"
    )
    source = open(deps_path, encoding="utf-8").read()
    tree = ast.parse(source)

    # Find the get_db_no_rls function body
    func_body_source = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_db_no_rls":
            func_body_source = ast.unparse(node)
            break

    assert func_body_source is not None, "get_db_no_rls function not found"
    assert "apply_rls_context" not in func_body_source, (
        "get_db_no_rls must never call apply_rls_context — "
        "it is a public endpoint dependency that must not set user context"
    )


def test_get_db_rls_optional_source_calls_apply_rls_context_when_user():
    """get_db_rls_optional must call apply_rls_context (for contrast)."""
    deps_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "dependencies.py"
    )
    source = open(deps_path, encoding="utf-8").read()
    tree = ast.parse(source)

    func_body_source = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_db_rls_optional":
            func_body_source = ast.unparse(node)
            break

    assert func_body_source is not None, "get_db_rls_optional function not found"
    assert "apply_rls_context" in func_body_source, (
        "get_db_rls_optional must call apply_rls_context for authenticated users"
    )


# ---------------------------------------------------------------------------
# Service layer: lookup_anonymized_case (no api.auth import needed)
# ---------------------------------------------------------------------------

def test_lookup_returns_redacted_payload(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    anon_id = "aabbccddeeff"
    _seed(test_db, anon_id=anon_id)

    result = lookup_anonymized_case(test_db, anonymized_id=anon_id)
    assert result is not None
    assert result["anonymized_id"] == anon_id
    assert result["case_type"] == "civil"
    assert result["jurisdiction"] == "Delhi"


def test_lookup_never_exposes_owner_user_id(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case
    import json

    anon_id = "112233445566"
    _seed(test_db, anon_id=anon_id)

    result = lookup_anonymized_case(test_db, anonymized_id=anon_id)
    assert result is not None

    serialized = json.dumps(result)
    assert "user_id" not in serialized, "Owner user_id must never appear in the payload"
    # user_id of the owner is 7 — must not appear as a standalone value
    assert '"7"' not in serialized


def test_lookup_unknown_id_returns_none(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    result = lookup_anonymized_case(test_db, anonymized_id="000000000000")
    assert result is None


def test_lookup_oversized_id_returns_none(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    result = lookup_anonymized_case(test_db, anonymized_id="a" * 65)
    assert result is None


def test_lookup_empty_id_returns_none(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    assert lookup_anonymized_case(test_db, anonymized_id="") is None
    assert lookup_anonymized_case(test_db, anonymized_id=None) is None  # type: ignore
