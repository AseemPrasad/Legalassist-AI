import os
import sys
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
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from database import Base  # noqa: E402
from db.models import (  # noqa: E402
    User,
    Case,
    CaseStatus,
    CaseDocument,
    CaseTimeline,
    Attachment,
    DocumentType,
)
from services import case_anonymization  # noqa: E402
from services import export_builder  # noqa: E402


@pytest.fixture(scope="function")
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = testing_session_local()

    user = User(id=1, email="test@example.com")
    db.add(user)
    db.commit()

    yield db
    db.close()


@pytest.fixture(autouse=True)
def patch_sessions(test_db, monkeypatch):
    monkeypatch.setattr(case_anonymization, "SessionLocal", lambda: test_db)

    @contextmanager
    def mock_session_local():
        yield test_db

    monkeypatch.setattr(export_builder, "SessionLocal", mock_session_local)


def _seed_privacy_case(test_db):
    case = Case(
        id=1,
        user_id=1,
        case_number="CASE-99999",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Rahul Sharma v. City Council",
        created_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc),
    )
    document = CaseDocument(
        id=10,
        case_id=1,
        document_type=DocumentType.JUDGMENT,
        document_content="Judgment text",
        uploaded_at=datetime(2026, 5, 2, 10, 0, 0, tzinfo=timezone.utc),
        summary="Call Rahul at test@example.com or +1 415 555 2671.",
        remedies={"next_step": "appeal"},
        extracted_metadata={"source": "court"},
        extraction_method="manual",
        ocr_used=False,
    )
    timeline = CaseTimeline(
        id=11,
        case_id=1,
        event_type="reminder",
        event_date=datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc),
        description="Reminder for Rahul Sharma at test@example.com",
        event_metadata={"message_preview": "Please call +1 415 555 2671"},
    )
    attachment = Attachment(
        id=12,
        user_id=1,
        case_id=1,
        original_filename="Rahul_Sharma_Affidavit.pdf",
        stored_path="/tmp/Rahul_Sharma_Affidavit.pdf",
        content_type="application/pdf",
        size_bytes=2048,
    )
    test_db.add_all([case, document, timeline, attachment])
    test_db.commit()


def test_personal_identifiers_profile_masks_direct_identifiers(test_db):
    _seed_privacy_case(test_db)

    data = case_anonymization.generate_anonymized_case_data(1, profile_name="personal_identifiers")

    assert data is not None
    assert data["privacy_profile"] == "personal_identifiers"
    assert data["privacy_profile_label"] == "Personal identifiers only"
    assert data["case_type"] == "civil"
    assert data["jurisdiction"] == "Delhi"
    assert data["documents"][0]["summary"] is not None
    assert "test@example.com" not in data["documents"][0]["summary"]
    assert "+1 415 555 2671" not in data["documents"][0]["summary"]
    assert "test@example.com" not in data["timeline"][0]["description"]
    assert "+1 415 555 2671" not in data["timeline"][0]["metadata"]["message_preview"]
    assert data["documents"][0]["remedies"] == {"next_step": "appeal"}


def test_full_party_removal_profile_redacts_narrative_fields(test_db):
    _seed_privacy_case(test_db)

    data = case_anonymization.generate_anonymized_case_data(1, profile_name="full_party_removal")

    assert data is not None
    assert data["privacy_profile"] == "full_party_removal"
    assert data["privacy_profile_label"] == "Full party removal"
    assert data["case_type"] == "civil"
    assert data["jurisdiction"] == "Delhi"
    assert data["documents"][0]["summary"] is None
    assert data["timeline"][0]["description"] is None
    assert data["documents"][0]["remedies"] is None
    assert data["attachments"][0]["original_filename"] is None


def test_export_builder_applies_privacy_profile_by_default(test_db):
    _seed_privacy_case(test_db)

    payload = export_builder.build_case_export_payload(
        user_id=1,
        case_id=1,
        field_ids=["case_number", "title", "documents", "timeline", "attachments", "remedies"],
        privacy_profile="full_party_removal",
    )

    assert payload is not None
    assert payload["export"]["privacy_profile"] == "full_party_removal"
    assert payload["case"]["case_number"] == "REDACTED"
    assert payload["case"]["title"] == "Redacted Matter"
    assert payload["documents"][0]["summary"] is None
    assert payload["timeline"][0]["description"] is None
    assert payload["attachments"][0]["original_filename"] is None
    assert payload["remedies"] is None
