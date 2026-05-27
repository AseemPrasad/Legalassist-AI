"""Tests for the anonymized case sharing and lookup feature.

Covers:
- GET /api/v1/anonymized-cases/{anonymized_id} happy path
- 404 for unknown / malformed IDs
- PII / owner identity never exposed
- Audit log entries recorded for both success and not-found lookups
- lookup_anonymized_case() service layer
- Existing anonymization tests continue to pass (smoke-checked via import)

NOTE: This test module uses raw SQL inserts to avoid the pre-existing
"Multiple classes found for path 'User'" mapper conflict that arises when
database.py (which redefines ORM models with extend_existing=True) is
imported alongside db.models.cases in the same process.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Environment setup — must happen before any project imports
# ---------------------------------------------------------------------------
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")
os.environ.setdefault("CASE_ANONYMIZATION_SECRET", "a" * 40)

# Stub out optional heavy dependencies that are not installed in the test env.
_OPTIONAL_STUBS = (
    "streamlit",
    "pytesseract",
    "pdf2image",
    "jaeger_client",
    "opentelemetry",
    "opentelemetry.exporter",
    "opentelemetry.exporter.prometheus",
    "opentelemetry.exporter.jaeger",
    "opentelemetry.exporter.jaeger.thrift",
    "opentelemetry.sdk",
    "opentelemetry.sdk.trace",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.sdk.resources",
    "opentelemetry.trace",
    "opentelemetry.metrics",
)
for _mod in _OPTIONAL_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from datetime import datetime, timezone
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Import db.base.Base for create_all — avoids triggering the mapper conflict.
from db.base import Base as _Base


# ---------------------------------------------------------------------------
# Module-level engine — one fresh in-memory DB for this entire test module.
# ---------------------------------------------------------------------------
_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
try:
    _Base.metadata.create_all(bind=_ENGINE)
except Exception:
    pass  # Tables already exist — safe to ignore.

_SessionFactory = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=_ENGINE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_db() -> Generator[Session, None, None]:
    """Provide a clean session for each test, rolling back after each test."""
    connection = _ENGINE.connect()
    transaction = connection.begin()
    session = _SessionFactory(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _make_app(test_db: Session, user_id: str | None = None) -> FastAPI:
    """Build a minimal FastAPI app with the anonymized-cases router."""
    import api.routes.anonymized_cases as anon_route
    from api.dependencies import get_db_rls_optional
    from api.auth import get_current_user_optional

    class _FakeUser:
        def __init__(self, uid: str):
            self.user_id = uid
            self.email = "test@example.com"
            self.role = "user"

    app = FastAPI()
    app.include_router(anon_route.router)
    app.dependency_overrides[get_db_rls_optional] = lambda: test_db
    if user_id is None:
        app.dependency_overrides[get_current_user_optional] = lambda: None
    else:
        _user = _FakeUser(user_id)
        app.dependency_overrides[get_current_user_optional] = lambda: _user
    return app


@pytest.fixture()
def client(test_db: Session) -> Generator[TestClient, None, None]:
    yield TestClient(_make_app(test_db))


@pytest.fixture()
def authed_client(test_db: Session) -> Generator[TestClient, None, None]:
    yield TestClient(_make_app(test_db, user_id="42"))


def _seed_case(db: Session, anon_id: str = "abc123def456") -> int:
    """Insert a Case row with a pre-set anonymized_id using raw SQL.

    Returns the inserted case_id.  Raw SQL avoids the ORM mapper conflict
    caused by database.py redefining models with extend_existing=True.
    """
    now = datetime.now(timezone.utc).isoformat()
    result = db.execute(
        text(
            "INSERT INTO cases "
            "(user_id, case_number, case_type, jurisdiction, status, title, "
            " anonymized_id, created_at, updated_at, version) "
            "VALUES (:uid, :cn, :ct, :jur, :st, :title, :anon_id, :now, :now, 1)"
        ),
        {
            "uid": 7,
            "cn": "2024-CV-00099",
            "ct": "civil",
            "jur": "Delhi",
            "st": "active",
            "title": "Private Matter",
            "anon_id": anon_id,
            "now": now,
        },
    )
    db.commit()
    case_id = result.lastrowid

    db.execute(
        text(
            "INSERT INTO case_documents "
            "(case_id, document_type, summary, remedies, uploaded_at, ocr_used) "
            "VALUES (:cid, :dt, :summary, :remedies, :now, 0)"
        ),
        {
            "cid": case_id,
            "dt": "Judgment",
            "summary": "Plaintiff seeks damages for breach of contract.",
            "remedies": '["Monetary compensation"]',
            "now": now,
        },
    )
    db.execute(
        text(
            "INSERT INTO case_timeline "
            "(case_id, event_type, description, event_date, created_at) "
            "VALUES (:cid, :et, :desc, :now, :now)"
        ),
        {
            "cid": case_id,
            "et": "filing",
            "desc": "Initial complaint filed.",
            "now": now,
        },
    )
    db.commit()
    return case_id


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_get_anonymized_case_success(client: TestClient, test_db: Session) -> None:
    anon_id = "aabbccddeeff"
    _seed_case(test_db, anon_id=anon_id)

    with patch("api.routes.anonymized_cases.record_immutable_audit_event") as mock_audit:
        resp = client.get(f"/api/v1/anonymized-cases/{anon_id}")

    assert resp.status_code == 200
    payload = resp.json()

    assert payload["anonymized_id"] == anon_id
    assert payload["case_type"] == "civil"
    assert payload["jurisdiction"] == "Delhi"
    assert payload["status"] == "active"
    assert payload["document_count"] == 1
    assert len(payload["documents"]) == 1
    assert len(payload["timeline"]) == 1

    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args.kwargs
    assert call_kwargs["outcome"] == "success"
    assert call_kwargs["resource_id"] == anon_id


def test_get_anonymized_case_no_owner_identity(client: TestClient, test_db: Session) -> None:
    """Owner user_id must never appear in the response."""
    anon_id = "112233445566"
    _seed_case(test_db, anon_id=anon_id)

    with patch("api.routes.anonymized_cases.record_immutable_audit_event"):
        resp = client.get(f"/api/v1/anonymized-cases/{anon_id}")
    assert resp.status_code == 200

    body = resp.text
    assert '"user_id"' not in body
    assert "Private Matter" not in body  # title is redacted by privacy profile


def test_get_anonymized_case_documents_redacted(client: TestClient, test_db: Session) -> None:
    """Document content (raw text) must not be exposed."""
    anon_id = "deadbeef1234"
    case_id = _seed_case(test_db, anon_id=anon_id)

    now = datetime.now(timezone.utc).isoformat()
    test_db.execute(
        text(
            "INSERT INTO case_documents "
            "(case_id, document_type, document_content, summary, uploaded_at, ocr_used) "
            "VALUES (:cid, :dt, :content, :summary, :now, 0)"
        ),
        {
            "cid": case_id,
            "dt": "FIR",
            "content": "Sensitive raw content with PII: john.doe@example.com",
            "summary": "FIR summary",
            "now": now,
        },
    )
    test_db.commit()

    with patch("api.routes.anonymized_cases.record_immutable_audit_event"):
        resp = client.get(f"/api/v1/anonymized-cases/{anon_id}")
    assert resp.status_code == 200

    body = resp.text
    assert "Sensitive raw content" not in body
    assert "john.doe@example.com" not in body


def test_get_anonymized_case_authenticated_user(
    authed_client: TestClient, test_db: Session
) -> None:
    """Authenticated users can also look up shared cases."""
    anon_id = "cafebabe5678"
    _seed_case(test_db, anon_id=anon_id)

    with patch("api.routes.anonymized_cases.record_immutable_audit_event") as mock_audit:
        resp = authed_client.get(f"/api/v1/anonymized-cases/{anon_id}")

    assert resp.status_code == 200
    call_kwargs = mock_audit.call_args.kwargs
    assert call_kwargs["actor_user_id"] == 42


# ---------------------------------------------------------------------------
# 404 / invalid ID tests
# ---------------------------------------------------------------------------


def test_get_anonymized_case_not_found(client: TestClient, test_db: Session) -> None:
    with patch("api.routes.anonymized_cases.record_immutable_audit_event") as mock_audit:
        resp = client.get("/api/v1/anonymized-cases/000000000000")

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

    mock_audit.assert_called_once()
    call_kwargs = mock_audit.call_args.kwargs
    assert call_kwargs["outcome"] == "not_found"


def test_get_anonymized_case_malformed_id_returns_404(client: TestClient) -> None:
    """Non-hex or too-short IDs should return 404 (not 422)."""
    for bad_id in ["../etc/passwd", "short", "ZZZZZZZZZZZZ"]:
        resp = client.get(f"/api/v1/anonymized-cases/{bad_id}")
        assert resp.status_code == 404, f"Expected 404 for id={bad_id!r}, got {resp.status_code}"


def test_get_anonymized_case_empty_db(client: TestClient, test_db: Session) -> None:
    resp = client.get("/api/v1/anonymized-cases/aabbccddeeff")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Service layer tests (use raw SQL to avoid mapper conflict)
# ---------------------------------------------------------------------------


def test_lookup_anonymized_case_returns_none_for_unknown(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    result = lookup_anonymized_case(test_db, anonymized_id="000000000000")
    assert result is None


def test_lookup_anonymized_case_returns_payload(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    anon_id = "f1e2d3c4b5a6"
    _seed_case(test_db, anon_id=anon_id)

    result = lookup_anonymized_case(test_db, anonymized_id=anon_id)
    assert result is not None
    assert result["anonymized_id"] == anon_id
    assert result["case_type"] == "civil"
    assert result["jurisdiction"] == "Delhi"
    assert "documents" in result
    assert "timeline" in result


def test_lookup_anonymized_case_rejects_oversized_id(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    result = lookup_anonymized_case(test_db, anonymized_id="a" * 65)
    assert result is None


def test_lookup_anonymized_case_rejects_empty_id(test_db: Session) -> None:
    from services.anonymized_case_lookup import lookup_anonymized_case

    assert lookup_anonymized_case(test_db, anonymized_id="") is None
    assert lookup_anonymized_case(test_db, anonymized_id=None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# anonymized_id column tests (raw SQL)
# ---------------------------------------------------------------------------


def test_case_model_has_anonymized_id_column(test_db: Session) -> None:
    """The cases table must have an anonymized_id column."""
    now = datetime.now(timezone.utc).isoformat()
    test_db.execute(
        text(
            "INSERT INTO cases "
            "(user_id, case_number, case_type, jurisdiction, status, "
            " anonymized_id, created_at, updated_at, version) "
            "VALUES (1, 'TEST-001', 'civil', 'Delhi', 'active', "
            "'testanon1234', :now, :now, 1)"
        ),
        {"now": now},
    )
    test_db.commit()

    row = test_db.execute(
        text("SELECT anonymized_id FROM cases WHERE anonymized_id = 'testanon1234'")
    ).fetchone()
    assert row is not None
    assert row[0] == "testanon1234"


def test_case_anonymized_id_unique_constraint(test_db: Session) -> None:
    """Two cases cannot share the same anonymized_id."""
    import sqlalchemy.exc

    now = datetime.now(timezone.utc).isoformat()
    test_db.execute(
        text(
            "INSERT INTO cases "
            "(user_id, case_number, case_type, jurisdiction, status, "
            " anonymized_id, created_at, updated_at, version) "
            "VALUES (1, 'UNIQ-001', 'civil', 'Delhi', 'active', "
            "'uniqueid12ab', :now, :now, 1)"
        ),
        {"now": now},
    )
    test_db.commit()

    with pytest.raises((sqlalchemy.exc.IntegrityError, Exception)):
        test_db.execute(
            text(
                "INSERT INTO cases "
                "(user_id, case_number, case_type, jurisdiction, status, "
                " anonymized_id, created_at, updated_at, version) "
                "VALUES (2, 'UNIQ-002', 'criminal', 'Mumbai', 'active', "
                "'uniqueid12ab', :now, :now, 1)"
            ),
            {"now": now},
        )
        test_db.commit()


# ---------------------------------------------------------------------------
# Smoke-check: existing anonymization tests still importable
# ---------------------------------------------------------------------------


def test_existing_anonymization_module_importable() -> None:
    """Ensure services.case_anonymization still imports without error."""
    import services.case_anonymization as ca  # noqa: F401

    assert callable(ca._generate_anonymized_case_id)
    assert callable(ca.generate_anonymized_case_data)
