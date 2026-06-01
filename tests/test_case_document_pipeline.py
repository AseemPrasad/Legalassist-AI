import os

if not os.environ.get("REDIS_URL"):
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
if not os.environ.get("CELERY_BROKER_URL"):
    os.environ["CELERY_BROKER_URL"] = os.environ["REDIS_URL"]
if not os.environ.get("CELERY_RESULT_BACKEND"):
    os.environ["CELERY_RESULT_BACKEND"] = os.environ["REDIS_URL"]
os.environ["JWT_SECRET_KEY"] = "test-secret-key-value-12345"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.routes.cases as cases_route
from api.auth import CurrentUser, get_current_user
from celery_app import process_case_document_upload_task
from database import Base, Case, CaseDocument, Attachment, CaseStatus, DocumentType


@pytest.fixture()
def test_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture()
def client(test_db, monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(cases_route.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("42", "tester@example.com", "user")
    app.dependency_overrides[cases_route.get_db] = lambda: test_db

    monkeypatch.setattr("core.storage.ATTACHMENTS_DIR", tmp_path)
    monkeypatch.setattr(cases_route, "validate_file_upload", lambda *args, **kwargs: None)

    async def fake_streaming_upload(*args, **kwargs):
        return 32

    monkeypatch.setattr(cases_route, "validate_file_upload_streaming", fake_streaming_upload)
    monkeypatch.setattr(
        cases_route,
        "enqueue_task_from_http_request",
        lambda *args, **kwargs: SimpleNamespace(id="job-123"),
    )
    return TestClient(app)


def _seed_case(test_db):
    case = Case(
        user_id=42,
        case_number="2024-CV-0001",
        title="Pipeline Case",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
    )
    test_db.add(case)
    test_db.commit()
    test_db.refresh(case)
    return case


def test_upload_case_document_endpoint_queues_job_and_links_attachment(client, test_db):
    case = _seed_case(test_db)

    response = client.post(
        f"/api/v1/cases/{case.id}/documents/upload",
        files={"file": ("petition.pdf", b"%PDF-1.4\ncase document\n", "application/pdf")},
        data={"document_type": "Judgment"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["task_id"] == "job-123"
    assert payload["attachment_id"]
    assert payload["document_id"]

    doc = test_db.query(CaseDocument).filter(CaseDocument.id == payload["document_id"]).first()
    att = test_db.query(Attachment).filter(Attachment.id == payload["attachment_id"]).first()

    assert doc is not None
    assert att is not None
    assert doc.source_attachment_id == att.id
    assert att.document_id == doc.id
    assert doc.extraction_method == "queued"
    assert doc.extracted_metadata == {"status": "queued"}


def test_process_case_document_upload_task_persists_metadata(test_db, tmp_path, monkeypatch):
    case = _seed_case(test_db)
    stored_path = tmp_path / "petition.pdf"
    stored_path.write_bytes(b"%PDF-1.4\nFake PDF text for OCR\n")

    attachment = Attachment(
        user_id=42,
        case_id=case.id,
        original_filename="petition.pdf",
        stored_path=str(stored_path),
        content_type="application/pdf",
        size_bytes=stored_path.stat().st_size,
    )
    test_db.add(attachment)
    test_db.commit()
    test_db.refresh(attachment)

    document = CaseDocument(
        case_id=case.id,
        source_attachment_id=attachment.id,
        document_type=DocumentType.JUDGMENT,
        file_path=str(stored_path),
        extraction_method="queued",
        ocr_used=False,
        extracted_metadata={"status": "queued"},
    )
    test_db.add(document)
    test_db.commit()
    test_db.refresh(document)

    monkeypatch.setattr(
        "celery_app.extract_text_from_uploaded_file",
        lambda *args, **kwargs: {
            "text": "John Doe v. State of Delhi filed on 12 March 2024 under Section 420 IPC.",
            "method": "ocr_tesseract",
            "ocr_used": True,
            "confidence": 91.5,
        },
    )
    monkeypatch.setattr(
        "celery_app.extract_case_document_metadata",
        lambda text, filename=None: {
            "title_hint": "John Doe v. State of Delhi",
            "parties": ["John Doe", "State of Delhi"],
            "dates": ["12 March 2024"],
            "claims": ["Compensation and injunctive relief sought."],
            "statutes": ["Section 420 IPC"],
            "confidence": {"parties": 0.9, "dates": 0.9, "claims": 0.8, "statutes": 0.95},
        },
    )

    result = process_case_document_upload_task.run(
        user_id="42",
        case_id=str(case.id),
        attachment_id=str(attachment.id),
        document_id=str(document.id),
        original_filename="petition.pdf",
    )

    assert result["status"] == "completed"
    assert result["ocr_used"] is True
    assert result["parties"] == ["John Doe", "State of Delhi"]

    updated_doc = test_db.query(CaseDocument).filter(CaseDocument.id == document.id).first()
    updated_attachment = test_db.query(Attachment).filter(Attachment.id == attachment.id).first()

    assert updated_doc.document_content.startswith("John Doe v. State of Delhi")
    assert updated_doc.ocr_used is True
    assert updated_doc.extraction_method == "ocr_tesseract"
    assert updated_doc.extracted_metadata["parties"] == ["John Doe", "State of Delhi"]
    assert updated_attachment.document_id == updated_doc.id
