import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Assuming api.main exports 'app'
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

@patch("api.auth.verify_token")
@patch("celery_app.TaskStatus.get_task_status")
def test_websocket_auth_via_subprotocol(mock_get_task_status, mock_verify_token, client):
    """Test that websocket connects successfully with valid token in subprotocols."""
    mock_verify_token.return_value = {"sub": "user_123"}
    # Mock task status to return completed so the loop exits immediately
    mock_get_task_status.return_value = {
        "status": "completed",
        "info": {"progress": 100},
        "timestamp": "2023-01-01T00:00:00Z"
    }

    token = "valid_token"
    job_id = "job_123"
    
    with client.websocket_connect(f"/ws/progress/{job_id}", subprotocols=["access_token", token]) as websocket:
        data = websocket.receive_json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"

@patch("api.auth.verify_token")
@patch("celery_app.TaskStatus.get_task_status")
def test_websocket_auth_via_query(mock_get_task_status, mock_verify_token, client):
    """Test that websocket connects successfully with valid token in query parameter (backward compatibility)."""
    mock_verify_token.return_value = {"sub": "user_123"}
    # Mock task status to return completed so the loop exits immediately
    mock_get_task_status.return_value = {
        "status": "completed",
        "info": {"progress": 100},
        "timestamp": "2023-01-01T00:00:00Z"
    }

    token = "valid_token"
    job_id = "job_123"
    
    with client.websocket_connect(f"/ws/progress/{job_id}?token={token}") as websocket:
        data = websocket.receive_json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"

def test_websocket_auth_missing_token(client):
    """Test that websocket rejects connection when no token is provided."""
    job_id = "job_123"
    try:
        with client.websocket_connect(f"/ws/progress/{job_id}") as websocket:
            pass
        pytest.fail("Should have rejected connection")
    except Exception as e:
        # TestClient raises WebSocketDisconnect on rejection
        assert hasattr(e, "code") or "4001" in str(e) or "403" in str(e) or type(e).__name__ == "WebSocketDisconnect"

@patch("api.auth.verify_token")
def test_websocket_auth_invalid_token(mock_verify_token, client):
    """Test that websocket rejects connection when invalid token is provided."""
    from api.auth import InvalidTokenError
    mock_verify_token.side_effect = InvalidTokenError("Invalid token")
    
    token = "invalid_token"
    job_id = "job_123"
    try:
        with client.websocket_connect(f"/ws/progress/{job_id}", subprotocols=["access_token", token]) as websocket:
            pass
        pytest.fail("Should have rejected connection")
    except Exception as e:
        assert hasattr(e, "code") or "4001" in str(e) or "403" in str(e) or type(e).__name__ == "WebSocketDisconnect"
