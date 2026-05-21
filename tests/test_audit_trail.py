import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

os.environ.setdefault("JWT_SECRET", "test-secret-key-that-is-long-enough")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost,127.0.0.1")
os.environ.setdefault("CASE_ANONYMIZATION_SECRET", "a" * 32)
sys.modules["streamlit"] = MagicMock()
sys.modules["pytesseract"] = MagicMock()
sys.modules["pdf2image"] = MagicMock()

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from database import Base  # noqa: E402
from db.models import User, Case, CaseStatus, CaseDocument, DocumentType  # noqa: E402
from db.crud.audit import record_audit_event, list_audit_events  # noqa: E402
from services.case_queries import get_case_detail  # noqa: E402
from services.case_anonymization import generate_anonymized_case_data  # noqa: E402
from services import export_builder  # noqa: E402
from auth import verify_otp_and_create_token, _hash_otp  # noqa: E402
from api.auth import CurrentUser  # noqa: E402
from api.routes.audit import get_case_audit_events
from api.auth import get_admin_user


@pytest.fixture(scope="function")
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = testing_session_local()
    yield db
    db.close()


@pytest.fixture(autouse=True)
def patch_sessions(test_db, monkeypatch):
    class DummySession:
        def __init__(self, db):
            self._db = db

        def __getattr__(self, item):
            return getattr(self._db, item)

        def close(self):
            return None

    monkeypatch.setattr("auth.SessionLocal", lambda: DummySession(test_db))
    monkeypatch.setattr("services.case_anonymization.SessionLocal", lambda: DummySession(test_db))

    @contextmanager
    def export_session_local():
        yield test_db

    monkeypatch.setattr(export_builder, "SessionLocal", export_session_local)


def _seed_user_and_case(test_db, *, user_id: int = 1, admin: bool = False):
    user = User(id=user_id, email=f"user{user_id}@example.com", is_admin=admin)
    test_db.add(user)
    case = Case(
        id=100,
        user_id=user_id,
        case_number="CASE-100",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Sensitive Case",
        created_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc),
    )
    doc = CaseDocument(
        id=200,
        case_id=100,
        document_type=DocumentType.JUDGMENT,
        document_content="Full text",
        summary="Call user@example.com immediately",
        remedies={"next_step": "appeal"},
        uploaded_at=datetime(2026, 5, 2, 10, 0, 0, tzinfo=timezone.utc),
    )
    test_db.add_all([case, doc])
    test_db.commit()
    return user, case, doc


def test_audit_event_sanitizes_sensitive_metadata(test_db):
    user, case, _doc = _seed_user_and_case(test_db)

    event = record_audit_event(
        test_db,
        actor=f"user:{user.id}",
        actor_user_id=user.id,
        action="download_report",
        resource=f"case:{case.id}",
        case_id=case.id,
        metadata={
            "email": "user@example.com",
            "password": "super-secret",
            "summary": "Contact user@example.com or +1 415 555 2671",
            "nested": {"token": "abc123", "note": "safe text"},
        },
    )

    assert event.event_metadata["email"] == "[redacted]"
    assert event.event_metadata["password"] == "[redacted]"
    assert "user@example.com" not in event.event_metadata["summary"]
    assert "+1 415 555 2671" not in event.event_metadata["summary"]
    assert event.event_metadata["nested"]["token"] == "[redacted]"
    assert event.event_metadata["nested"]["note"] == "safe text"


def test_login_success_records_audit_event(test_db, monkeypatch):
    user, _case, _doc = _seed_user_and_case(test_db)

    otp_hash = _hash_otp("123456")
    fake_otp_hash = otp_hash

    class FakeOtp:
        id = 7
        otp_hash = fake_otp_hash
        failed_attempts = 0
        locked_until = None

        def is_locked(self):
            return False

    monkeypatch.setattr("auth.get_pending_otp", lambda db, email: FakeOtp())
    monkeypatch.setattr("auth.reset_otp_failed_attempts", lambda db, otp_id: None)
    monkeypatch.setattr("auth.mark_otp_as_used", lambda db, otp_id: None)
    monkeypatch.setattr("auth.update_user_last_login", lambda db, user_id: None)
    monkeypatch.setattr("auth.get_user_by_email", lambda db, email: user)
    monkeypatch.setattr("auth.create_user", lambda db, email: user)
    monkeypatch.setattr("auth.create_jwt_token", lambda user_id, email: "jwt-token")

    success, message, token = verify_otp_and_create_token("user1@example.com", "123456")

    assert success is True
    assert message == "Login successful"
    assert token == "jwt-token"

    events = list_audit_events(test_db, actor_user_id=user.id)
    assert any(event.action == "login_success" for event in events)


def test_case_view_and_privacy_actions_write_audit_events(test_db):
    user, case, _doc = _seed_user_and_case(test_db)

    detail = get_case_detail(test_db, user_id=user.id, case_id=case.id)
    assert detail is not None

    anonymized = generate_anonymized_case_data(case.id, profile_name="full_party_removal")
    assert anonymized is not None

    export_artifact = export_builder.build_case_export_artifact(
        user_id=user.id,
        case_id=case.id,
        format="json",
        field_ids=["case_number", "title", "documents"],
        privacy_profile="personal_identifiers",
    )
    assert export_artifact is not None

    actions = [event.action for event in list_audit_events(test_db, case_id=case.id)]
    assert "view_case_detail" in actions
    assert "anonymization_run" in actions
    assert "export_download" in actions


def test_case_audit_access_control_enforced(test_db):
    owner, case, _doc = _seed_user_and_case(test_db)
    other_user = User(id=2, email="other@example.com", is_admin=False)
    admin_user = User(id=3, email="admin@example.com", is_admin=True)
    test_db.add_all([other_user, admin_user])
    test_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        import asyncio
        asyncio.run(
            get_case_audit_events(
                case_id=case.id,
                limit=10,
                db=test_db,
                current_user=CurrentUser(user_id=other_user.id, email=other_user.email, role="user"),
            )
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info_2:
        asyncio.run(get_admin_user(CurrentUser(user_id=owner.id, email=owner.email, role="user")))
    assert exc_info_2.value.status_code == 403

    allowed = asyncio.run(get_admin_user(CurrentUser(user_id=admin_user.id, email=admin_user.email, role="admin")))
    assert allowed.role == "admin"
