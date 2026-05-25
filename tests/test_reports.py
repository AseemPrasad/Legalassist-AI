import os
import sys
import types
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-value-12345")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

if "report_service" not in sys.modules:
    report_service_stub = types.ModuleType("report_service")
    report_service_stub._get_reports_base_dir = lambda: None
    sys.modules["report_service"] = report_service_stub

from api.auth import CurrentUser, get_current_user
from api.routes.reports import router, get_db
from database import Base, Report
from celery_app import TaskStatus


@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory test database for report testing"""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="function")
def client(test_db):
    """FastAPI TestClient with overridden DB and Auth dependencies"""
    app = FastAPI()
    app.include_router(router)
    
    # Override get_db dependency
    app.dependency_overrides[get_db] = lambda: test_db
    # Override get_current_user dependency
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(42, "tester@example.com", "user")
    
    return TestClient(app)


def test_generate_report_flow(client, test_db, monkeypatch):
    """Test generating a report: creates DB record and enqueues task correctly"""
    mock_task = MagicMock()
    mock_task.id = "mock-celery-job-123"
    
    monkeypatch.setattr(
        "api.routes.reports.enqueue_task_from_http_request",
        lambda *args, **kwargs: mock_task
    )

    payload = {
        "case_id": "CASE-999",
        "report_type": "comprehensive",
        "format": "pdf"
    }
    response = client.post("/api/v1/reports/generate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "CASE-999"
    assert data["job_id"] == "mock-celery-job-123"
    assert data["status"] == "pending"
    assert "report_id" in data

    # Verify database persistence
    db_report = test_db.query(Report).filter(Report.report_id == data["report_id"]).first()
    assert db_report is not None
    assert db_report.user_id == 42
    assert db_report.case_id == "CASE-999"
    assert db_report.job_id == "mock-celery-job-123"
    assert db_report.status == "pending"


def test_generate_report_flow_reuses_single_report_id(monkeypatch):
    """POST /api/v1/reports/generate reuses the same report_id for DB and enqueue flow."""
    app = FastAPI()
    app.include_router(router)

    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(42, "tester@example.com", "user")
    client = TestClient(app)

    mock_task = MagicMock()
    mock_task.id = "mock-celery-job-456"
    captured_kwargs = {}

    mock_report = MagicMock(spec=Report)

    def fake_enqueue(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_task

    def fake_create_report(*args, **kwargs):
        mock_report.report_id = kwargs["report_id"]
        mock_report.user_id = kwargs["user_id"]
        mock_report.case_id = kwargs["case_id"]
        mock_report.celery_task_id = kwargs["celery_task_id"]
        mock_report.report_type = kwargs["report_type"]
        mock_report.format = kwargs["format"]
        mock_report.style = kwargs["style"]
        mock_report.status = "pending"
        mock_report.created_at = datetime.now(timezone.utc)
        return mock_report

    monkeypatch.setattr(
        "api.routes.reports.enqueue_task_from_http_request",
        fake_enqueue,
    )
    monkeypatch.setattr("api.routes.reports.create_report", fake_create_report)
    monkeypatch.setattr("api.routes.reports.update_report_status", lambda *args, **kwargs: None)
    mock_db.query.return_value.filter.return_value.first.return_value = mock_report

    payload = {
        "case_id": "100",
        "report_type": "comprehensive",
        "format": "pdf",
    }
    response = client.post("/api/v1/reports/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    report_id = data["report_id"]
    assert captured_kwargs["report_id"] == report_id

    assert mock_report.report_id == report_id
    assert mock_report.job_id == mock_task.id
    assert mock_report.celery_task_id == mock_task.id

    generated_file_name = f"{mock_report.case_id}_{mock_report.report_type}_{report_id}.pdf"
    assert generated_file_name.endswith(f"{report_id}.pdf")
    assert report_id in generated_file_name


def test_get_report_status_not_found(client):
    """GET /api/v1/reports/{report_id} returns 404 if not found"""
    response = client.get("/api/v1/reports/non-existent-uuid")
    assert response.status_code == 404
    assert response.json()["detail"] == "Report not found"


def test_get_report_status_updates_from_celery(client, test_db, monkeypatch):
    """GET /api/v1/reports/{report_id} resolves status from Celery and updates DB"""
    # Create database record manually
    db_report = Report(
        report_id="test-report-uuid-456",
        job_id="celery-job-uuid-456",
        user_id=42,
        case_id="CASE-123",
        format="pdf",
        status="pending"
    )
    test_db.add(db_report)
    test_db.commit()

    # Mock Celery status check
    mock_status_info = {
        "task_id": "celery-job-uuid-456",
        "status": "completed",
        "info": {},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    monkeypatch.setattr(
        TaskStatus,
        "get_task_status",
        lambda job_id: mock_status_info
    )

    response = client.get("/api/v1/reports/test-report-uuid-456")
    assert response.status_code == 200
    data = response.json()
    assert data["report_id"] == "test-report-uuid-456"
    assert data["status"] == "completed"
    assert data["download_url"] == "/api/v1/reports/test-report-uuid-456/download"

    # Verify DB has been updated to completed
    test_db.refresh(db_report)
    assert db_report.status == "completed"
    assert db_report.completed_at is not None


def test_download_report_security_and_file_path(client, test_db, monkeypatch, tmp_path):
    """GET /api/v1/reports/{report_id}/download checks ownership and returns file if ready"""
    # Create database record manually with different user ownership (user 99)
    db_report_other = Report(
        report_id="report-other-user",
        job_id="job-other",
        user_id=99,
        case_id="CASE-123",
        format="pdf",
        status="completed"
    )
    test_db.add(db_report_other)
    test_db.commit()

    # Unauthorized access (user 42 requests user 99's report)
    response = client.get("/api/v1/reports/report-other-user/download")
    assert response.status_code == 404

    # Authorized report owned by user 42
    db_report_own = Report(
        report_id="report-own-user",
        job_id="job-own",
        user_id=42,
        case_id="CASE-123",
        format="pdf",
        status="completed"
    )
    test_db.add(db_report_own)
    test_db.commit()

    # Mock _get_reports_base_dir to use a temp path
    user_dir = tmp_path / "42"
    user_dir.mkdir(parents=True, exist_ok=True)
    report_file = user_dir / "CASE-123_comprehensive_report-own-user.pdf"
    report_file.write_text("dummy PDF content")

    monkeypatch.setattr(
        "api.routes.reports._get_reports_base_dir",
        lambda: tmp_path
    )

    response = client.get("/api/v1/reports/report-own-user/download")
    assert response.status_code == 200
    assert response.content == b"dummy PDF content"


def test_list_reports_pagination(client, test_db):
    """GET /api/v1/reports lists, paginates, and orders reports correctly"""
    # Create multiple database records
    reports = [
        Report(
            report_id=f"rep-{i}",
            job_id=f"job-{i}",
            user_id=42,
            case_id="CASE-123",
            format="pdf",
            status="completed" if i % 2 == 0 else "pending",
            created_at=datetime.now(timezone.utc)
        )
        for i in range(1, 6)
    ]
    test_db.add_all(reports)
    test_db.commit()

    response = client.get("/api/v1/reports?limit=2&offset=1")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["reports"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 1
