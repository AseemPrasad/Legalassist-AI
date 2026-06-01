"""
Comprehensive tests for Report Generation API endpoints.

Tests cover:
- POST /generate: Report creation and task enqueueing
- GET /{report_id}: Status tracking via DB and Celery
- GET /{report_id}/download: Ownership validation, status checks, file access
- GET /: Report listing
- Unauthorized download attempts
- Completed vs pending behavior
"""

import pytest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from tempfile import TemporaryDirectory

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.reports as reports_route
from api.auth import CurrentUser, get_current_user
from api.models import ReportGenerationRequest, ReportGenerationResponse
from db.models import Report, ReportStatus, ReportType, ReportFormat


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def app():
    """Create FastAPI test app with reports router."""
    app = FastAPI()
    app.include_router(reports_route.router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_current_user():
    """Create a mock current user."""
    return CurrentUser(user_id=42, email="test@example.com", role="user")


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def mock_report():
    """Create a mock Report model instance."""
    report = Mock(spec=Report)
    report.id = 1
    report.report_id = str(uuid.uuid4())
    report.user_id = 42
    report.case_id = 100
    report.job_id = "celery-task-id-123"
    report.status = ReportStatus.PENDING
    report.report_type = ReportType.COMPREHENSIVE
    report.format = ReportFormat.PDF
    report.created_at = datetime.utcnow()
    report.completed_at = None
    report.case = Mock()
    report.user = Mock()
    return report


@pytest.fixture
def completed_report():
    """Create a mock completed Report."""
    report = Mock(spec=Report)
    report.id = 2
    report.report_id = str(uuid.uuid4())
    report.user_id = 42
    report.case_id = 100
    report.job_id = "celery-task-id-456"
    report.status = ReportStatus.COMPLETED
    report.report_type = ReportType.COMPREHENSIVE
    report.format = ReportFormat.PDF
    report.style = "formal"
    report.file_path = "/reports/42/case_100_comprehensive_report.pdf"
    report.file_size_bytes = 524288
    report.error_message = None
    report.created_at = datetime.utcnow()
    report.started_at = datetime.utcnow()
    report.completed_at = datetime.utcnow()
    report.updated_at = datetime.utcnow()
    report.case = Mock()
    report.user = Mock()
    return report


# ============================================================================
# Test POST /generate
# ============================================================================

def test_generate_report_creates_db_record_and_enqueues_task(
    monkeypatch, app, client, mock_current_user, mock_db, mock_report
):
    """Test POST /generate creates Report record and enqueues Celery task."""
    
    # Setup mocks
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    mock_task = Mock()
    mock_task.id = "celery-task-id-123"
    
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    def mock_create_report(**kwargs):
        mock_report.report_id = kwargs["report_id"]
        mock_report.job_id = "pending"
        return mock_report
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "create_report", mock_create_report)
    monkeypatch.setattr(reports_route, "enqueue_task_from_http_request", lambda *a, **kw: mock_task)
    monkeypatch.setattr(reports_route, "update_report_status", lambda *a, **kw: None)
    
    request_data = {
        "case_id": "100",
        "report_type": "comprehensive",
        "format": "pdf",
        "style": "formal",
        "include_remedies": True,
        "include_timeline": True,
        "include_similar_cases": True,
    }
    
    response = client.post("/api/v1/reports/generate", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["report_type"] == "comprehensive"
    assert data["format"] == "pdf"
    assert "report_id" in data
    assert "job_id" in data


def test_generate_report_invalid_case_id(
    monkeypatch, app, client, mock_current_user, mock_db
):
    """Test POST /generate rejects invalid case_id."""
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    
    request_data = {
        "case_id": "not-a-number",
        "report_type": "comprehensive",
        "format": "pdf",
    }
    
    response = client.post("/api/v1/reports/generate", json=request_data)
    
    assert response.status_code == 400
    assert "Invalid case_id" in response.json()["detail"]


# ============================================================================
# Test GET /{report_id}
# ============================================================================

def test_get_report_status_returns_db_data(
    monkeypatch, app, client, mock_current_user, mock_db, mock_report
):
    """Test GET /{report_id} returns report status from DB using stored celery_task_id."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    # Mock TaskStatus.get_task_status to return task status
    mock_task_status = {
        "task_id": "celery-task-id-123",
        "status": "processing",
        "info": {"progress": 50},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: mock_report)
    monkeypatch.setattr(reports_route.TaskStatus, "get_task_status", lambda tid: mock_task_status)
    
    response = client.get(f"/api/v1/reports/{mock_report.report_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["job_id"] == "celery-task-id-123"  # Stored celery_task_id, not report_id


def test_get_report_status_unauthorized_access(
    monkeypatch, app, client, mock_current_user, mock_db
):
    """Test GET /{report_id} returns 404 for reports not belonging to user."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    # Mock returns None because report doesn't belong to user
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: None)
    
    response = client.get("/api/v1/reports/unknown-report-id")
    
    assert response.status_code == 404
    assert "Report not found" in response.json()["detail"]


def test_get_report_status_completed(
    monkeypatch, app, client, mock_current_user, mock_db, completed_report
):
    """Test GET /{report_id} returns completed status with download_url."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    mock_task_status = {
        "task_id": "celery-task-id-456",
        "status": "completed",
        "info": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: completed_report)
    monkeypatch.setattr(reports_route.TaskStatus, "get_task_status", lambda tid: mock_task_status)
    
    response = client.get(f"/api/v1/reports/{completed_report.report_id}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["download_url"] is not None
    assert data["file_size_bytes"] == 524288


# ============================================================================
# Test GET /{report_id}/download
# ============================================================================

def test_download_report_success(
    monkeypatch, app, client, mock_current_user, mock_db, completed_report
):
    """Test GET /{report_id}/download returns file for completed report."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    # Create a temporary file to return
    with TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_report.pdf"
        test_file.write_bytes(b"PDF content here")
        
        completed_report.file_path = str(test_file)
        
        monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
        monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: completed_report)
        
        response = client.get(f"/api/v1/reports/{completed_report.report_id}/download")
        
        assert response.status_code == 200
        assert response.content == b"PDF content here"


def test_download_report_pending_status_rejects(
    monkeypatch, app, client, mock_current_user, mock_db, mock_report
):
    """Test GET /{report_id}/download rejects pending reports."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    mock_report.status = ReportStatus.PENDING
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: mock_report)
    
    response = client.get(f"/api/v1/reports/{mock_report.report_id}/download")
    
    assert response.status_code == 202
    assert "still pending" in response.json()["detail"]


def test_download_report_unauthorized_user(
    monkeypatch, app, client, mock_current_user, mock_db, completed_report
):
    """Test GET /{report_id}/download rejects unauthorized user."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    # Report belongs to different user
    completed_report.user_id = 999
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: completed_report)
    
    response = client.get(f"/api/v1/reports/{completed_report.report_id}/download")
    
    assert response.status_code == 403
    assert "Not authorized" in response.json()["detail"]


def test_download_report_file_missing_on_disk(
    monkeypatch, app, client, mock_current_user, mock_db, completed_report
):
    """Test GET /{report_id}/download handles missing file on disk."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    # File path stored in DB but doesn't exist on disk
    completed_report.file_path = "/nonexistent/report.pdf"
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: completed_report)
    
    response = client.get(f"/api/v1/reports/{completed_report.report_id}/download")
    
    assert response.status_code == 404
    assert "file not found on disk" in response.json()["detail"]


def test_download_report_no_file_path_in_db(
    monkeypatch, app, client, mock_current_user, mock_db, completed_report
):
    """Test GET /{report_id}/download handles missing file_path in DB."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    # DB record has no file_path
    completed_report.file_path = None
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: completed_report)
    
    response = client.get(f"/api/v1/reports/{completed_report.report_id}/download")
    
    assert response.status_code == 404
    assert "file path not found" in response.json()["detail"]


# ============================================================================
# Test GET / (list reports)
# ============================================================================

def test_list_reports_returns_user_reports(
    monkeypatch, app, client, mock_current_user, mock_db, mock_report, completed_report
):
    """Test GET / returns paginated list of user's reports."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    reports = [mock_report, completed_report]
    total = len(reports)
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(
        reports_route,
        "list_reports_by_user",
        lambda db, user_id, limit, offset, status: (reports, total)
    )
    
    response = client.get("/api/v1/reports?limit=10&offset=0")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["limit"] == 10
    assert data["offset"] == 0
    assert len(data["reports"]) == 2


def test_list_reports_with_status_filter(
    monkeypatch, app, client, mock_current_user, mock_db, completed_report
):
    """Test GET / filters reports by status."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    reports = [completed_report]
    total = 1
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(
        reports_route,
        "list_reports_by_user",
        lambda db, user_id, limit, offset, status: (reports, total) if status == "completed" else ([], 0)
    )
    
    response = client.get("/api/v1/reports?limit=10&offset=0&status_filter=completed")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1


# ============================================================================
# Integration-style tests
# ============================================================================

def test_report_workflow_create_and_check_status(
    monkeypatch, app, client, mock_current_user, mock_db, mock_report
):
    """Integration test: Create report and check status progression."""
    
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    
    mock_task = Mock()
    mock_task.id = "celery-task-id-123"
    
    # Mock create_report to return our mock_report
    def mock_create_report_fn(**kwargs):
        mock_report.report_id = kwargs["report_id"]
        return mock_report
    
    # Setup mocks
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "create_report", mock_create_report_fn)
    monkeypatch.setattr(reports_route, "enqueue_task_from_http_request", lambda *a, **kw: mock_task)
    monkeypatch.setattr(reports_route, "update_report_status", lambda *a, **kw: None)
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: mock_report)
    monkeypatch.setattr(reports_route.TaskStatus, "get_task_status", 
                       lambda tid: {"task_id": tid, "status": "processing", "info": {}, "timestamp": ""})
    
    # Step 1: Create report
    request_data = {
        "case_id": "100",
        "report_type": "comprehensive",
        "format": "pdf",
    }
    create_response = client.post("/api/v1/reports/generate", json=request_data)
    assert create_response.status_code == 200
    report_id = create_response.json()["report_id"]
    
    # Step 2: Check status (should show processing)
    status_response = client.get(f"/api/v1/reports/{report_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "processing"


def test_report_ownership_isolation(
    monkeypatch, app, client, mock_current_user, mock_db, completed_report
):
    """Test that users cannot access each other's reports."""
    
    # User 1 tries to access User 2's report
    user1 = CurrentUser(user_id=42, email="user1@example.com", role="user")
    app.dependency_overrides[get_current_user] = lambda: user1
    
    completed_report.user_id = 99  # Report belongs to user 99
    
    monkeypatch.setattr(reports_route, "get_db", lambda: iter([mock_db]))
    monkeypatch.setattr(reports_route, "get_report_by_id", lambda db, rid, user_id: None)  # Not found for user 42
    
    response = client.get(f"/api/v1/reports/{completed_report.report_id}")
    
    assert response.status_code == 404
