from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.base import Base
from db.models import Case, NotificationChannel, NotificationLog, NotificationStatus, User
from db.crud.notifications import log_notification
from services.export_builder import build_case_export_artifact, build_case_export_payload


@pytest.fixture()
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = testing_session()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def session_factory(test_db):
    engine = test_db.get_bind()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


class TestPrivacyHardening:
    def test_notification_log_recipient_is_masked_on_persist(self, test_db):
        log = log_notification(
            db=test_db,
            deadline_id=101,
            user_id=7,
            channel=NotificationChannel.EMAIL,
            recipient="alice.smith@example.com",
            days_before=10,
            status=NotificationStatus.SENT,
            message_preview="Reminder for alice.smith@example.com",
        )

        assert log.recipient != "alice.smith@example.com"
        assert "alice.smith@example.com" not in log.recipient
        assert "@example.com" in log.recipient
        assert "alice" not in log.recipient
        assert "alice.smith@example.com" not in (log.message_preview or "")

    def test_export_payload_redacts_case_number_and_uses_safe_filename(self, test_db, session_factory, monkeypatch):
        user = User(email="owner@example.com")
        test_db.add(user)
        test_db.flush()

        case = Case(
            user_id=user.id,
            case_number="CASE-SECRET-001",
            case_type="civil",
            jurisdiction="Delhi",
            title="Confidential Matter",
        )
        test_db.add(case)
        test_db.commit()

        monkeypatch.setattr("services.export_builder.SessionLocal", session_factory)

        payload = build_case_export_payload(
            user_id=user.id,
            case_id=case.id,
            privacy_profile="personal_identifiers",
        )
        assert payload is not None
        assert payload["export"]["case_number"] != "CASE-SECRET-001"
        assert "CASE-SECRET-001" not in payload["export"]["case_number"]
        assert payload["case"]["case_number"] != "CASE-SECRET-001"
        assert "CASE-SECRET-001" not in payload["case"]["case_number"]

        artifact = build_case_export_artifact(
            user_id=user.id,
            case_id=case.id,
            format="json",
            privacy_profile="personal_identifiers",
        )
        assert artifact is not None
        assert artifact.file_name == f"case_{case.id}_export.json"
        assert "CASE-SECRET-001" not in artifact.file_name
