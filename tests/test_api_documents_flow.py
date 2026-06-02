import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-that-is-long-enough"
from api.main import app

client = TestClient(app)

@pytest.fixture
def override_auth():
    from api.auth import get_current_user, CurrentUser
    def override_get_current_user():
        return CurrentUser(user_id="test_user", email="test@example.com", role="user")
    app.dependency_overrides[get_current_user] = override_get_current_user
    yield
    app.dependency_overrides.clear()

def test_analyze_document_text(override_auth):
    with patch("api.routes.documents.enqueue_task_from_http_request") as mock_enqueue:
        mock_task = MagicMock()
        mock_task.id = "test-job-id"
        mock_enqueue.return_value = mock_task
        
        response = client.post("/api/v1/analyze/document", json={
            "text": "This is a test document.",
            "document_type": "contract"
        })
        
    assert response.status_code == 200
    assert response.json()["job_id"] == "test-job-id"
    assert response.json()["status"] == "pending"

def test_analyze_document_missing_content(override_auth):
    response = client.post("/api/v1/analyze/document", json={
        "document_type": "contract"
    })
    
    assert response.status_code == 400

def test_get_analysis_status_pending(override_auth):
    with patch("api.routes.documents.TaskStatus.get_task_status") as mock_status:
        mock_status.return_value = {"status": "pending", "info": {}, "timestamp": "now"}
        
        response = client.get("/api/v1/analyze/test-job-id")
        
    assert response.status_code == 200
    assert response.json()["status"] == "pending"

def test_get_analysis_status_completed(override_auth):
    with patch("api.routes.documents.TaskStatus.get_task_status") as mock_status:
        mock_status.return_value = {"status": "completed", "info": {}, "timestamp": "now"}
        
        response = client.get("/api/v1/analyze/test-job-id")
        
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert "result_url" in response.json()

def test_get_analysis_result_completed(override_auth):
    with patch("api.routes.documents.TaskStatus.get_task_status") as mock_status:
        mock_status.return_value = {
            "status": "completed", 
            "info": {
                "document_id": "doc-123",
                "title": "Test Title",
                "document_type": "contract"
            }, 
            "timestamp": "now"
        }
        
        response = client.get("/api/v1/analyze/test-job-id/result")
        
    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-123"

def test_get_analysis_result_pending(override_auth):
    with patch("api.routes.documents.TaskStatus.get_task_status") as mock_status:
        mock_status.return_value = {"status": "pending", "info": {}, "timestamp": "now"}
        
        response = client.get("/api/v1/analyze/test-job-id/result")
        
    assert response.status_code == 202

def test_cancel_analysis(override_auth):
    with patch("api.routes.documents.TaskStatus.revoke_task") as mock_revoke:
        mock_revoke.return_value = True
        
        response = client.post("/api/v1/analyze/test-job-id/cancel")
        
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

def test_upload_document(override_auth):
    with patch("api.routes.documents.enqueue_task_from_http_request") as mock_enqueue, \
         patch("api.routes.documents.validate_file_upload") as mock_validate, \
         patch("api.routes.documents.validate_file_upload_streaming") as mock_validate_stream:
         
        mock_task = MagicMock()
        mock_task.id = "test-job-id"
        mock_enqueue.return_value = mock_task
        
        # Async mock for validate_file_upload_streaming
        import asyncio
        future = asyncio.Future()
        future.set_result(100)
        mock_validate_stream.return_value = future
        
        files = {"file": ("test.txt", b"Test content", "text/plain")}
        data = {"document_type": "contract"}
        
        response = client.post("/api/v1/analyze/upload", files=files, data=data)
        
    assert response.status_code == 200
    assert response.json()["job_id"] == "test-job-id"
