import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock
from zipfile import ZipFile

os.environ.setdefault("JWT_SECRET", "test-secret-key-that-is-long-enough")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost,127.0.0.1")
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
    CaseDeadline,
    CaseDocument,
    CaseTimeline,
    Attachment,
    DocumentType,
)
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
def patch_session_local(test_db, monkeypatch):
    @contextmanager
    def mock_session_local():
        yield test_db

    monkeypatch.setattr(export_builder, "SessionLocal", mock_session_local)


def _seed_case(test_db, *, case_id: int, case_number: str, title: str) -> Case:
    case = Case(
        id=case_id,
        user_id=1,
        case_number=case_number,
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title=title,
    )
    test_db.add(case)
    test_db.commit()

    document = CaseDocument(
        id=case_id * 10,
        case_id=case_id,
        document_type=DocumentType.JUDGMENT,
        document_content="Judgment text",
        uploaded_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc),
        summary="Important ruling",
        remedies={"next_step": "appeal"},
        extracted_metadata={"source": "court"},
        extraction_method="manual",
        ocr_used=False,
    )
    deadline = CaseDeadline(
        id=case_id * 10 + 1,
        user_id=1,
        case_id=case_id,
        case_title=title,
        deadline_date=datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        deadline_type="appeal",
        description="File appeal",
        is_completed=False,
    )
    timeline = CaseTimeline(
        id=case_id * 10 + 2,
        case_id=case_id,
        event_type="document_uploaded",
        event_date=datetime(2026, 5, 1, 11, 0, 0, tzinfo=timezone.utc),
        description="Judgment uploaded",
        event_metadata={"document_id": document.id},
    )
    attachment = Attachment(
        id=case_id * 10 + 3,
        user_id=1,
        case_id=case_id,
        original_filename=f"{case_number}.pdf",
        stored_path=f"/tmp/{case_number}.pdf",
        content_type="application/pdf",
        size_bytes=1024,
    )

    test_db.add_all([document, deadline, timeline, attachment])
    test_db.commit()
    return case


def test_single_case_export_builder_formats(test_db):
    _seed_case(test_db, case_id=1, case_number="CASE-12345", title="My Civil Suit")

    payload = export_builder.build_case_export_payload(
        user_id=1,
        case_id=1,
        field_ids=["case_number", "title", "document_count", "latest_document", "next_deadline", "documents", "deadlines", "timeline", "attachments", "remedies"],
    )

    assert payload is not None
    assert payload["export"]["case_number"] == "CASE-12345"
    assert payload["case"]["case_number"] == "CASE-12345"
    assert payload["case"]["title"] == "My Civil Suit"
    assert payload["document_count"] == 1
    assert payload["latest_document"]["summary"] == "Important ruling"
    assert payload["next_deadline"]["deadline_type"] == "appeal"
    assert len(payload["documents"]) == 1
    assert len(payload["deadlines"]) == 1
    assert len(payload["timeline"]) == 1
    assert len(payload["attachments"]) == 1
    assert payload["remedies"] == {"next_step": "appeal"}

    json_artifact = export_builder.build_case_export_artifact(
        user_id=1,
        case_id=1,
        format="json",
        field_ids=["case_number", "title", "document_count", "latest_document", "next_deadline"],
    )
    pdf_artifact = export_builder.build_case_export_artifact(
        user_id=1,
        case_id=1,
        format="pdf",
        field_ids=["case_number", "title", "document_count", "latest_document", "next_deadline"],
    )
    docx_artifact = export_builder.build_case_export_artifact(
        user_id=1,
        case_id=1,
        format="docx",
        field_ids=["case_number", "title", "document_count", "latest_document", "next_deadline"],
    )

    assert json_artifact is not None
    assert pdf_artifact is not None
    assert docx_artifact is not None
    assert json_artifact.data.decode("utf-8").startswith("{\n  \"export\"")
    assert pdf_artifact.data.startswith(b"%PDF")

    with ZipFile(BytesIO(docx_artifact.data)) as archive:
        assert "word/document.xml" in archive.namelist()
        document_xml = archive.read("word/document.xml").decode("utf-8")
        assert "CASE-12345" in document_xml
        assert "My Civil Suit" in document_xml


def test_case_export_bundle_contains_manifest_and_stable_order(test_db):
    _seed_case(test_db, case_id=1, case_number="CASE-12345", title="My Civil Suit")
    _seed_case(test_db, case_id=2, case_number="CASE-67890", title="Another Matter")

    bundle = export_builder.build_case_export_bundle(
        user_id=1,
        case_ids=[1, 2],
        field_ids=["case_number", "title", "document_count"],
        formats=["json", "pdf"],
    )

    assert bundle is not None
    assert bundle.format == "zip"
    assert bundle.file_name.endswith(".zip")

    with ZipFile(BytesIO(bundle.data)) as archive:
        names = archive.namelist()
        assert names[0] == "case_1/CASE-12345_export.json"
        assert names[1] == "case_1/CASE-12345_export.pdf"
        assert names[2] == "case_2/CASE-67890_export.json"
        assert names[3] == "case_2/CASE-67890_export.pdf"
        manifest = archive.read("manifest.json").decode("utf-8")
        assert '"case_ids": [\n    1,\n    2\n  ]' in manifest
        assert '"formats": [\n    "json",\n    "pdf"\n  ]' in manifest
        assert '"fields": [\n    "case_number",\n    "title",\n    "document_count"\n  ]' in manifest
