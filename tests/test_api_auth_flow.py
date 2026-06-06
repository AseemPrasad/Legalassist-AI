import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
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

def test_get_token():
    response = client.post("/api/v1/auth/token", params={"username": "testuser", "password": "password"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_create_api_key(override_auth):
    with patch("api.routes.auth.create_api_key_record") as mock_create:
        mock_create.return_value = ("test-key-123", None)
        response = client.post("/api/v1/auth/api-keys", json={"name": "Test Key", "expires_in_days": 30})
        
    assert response.status_code == 200
    assert response.json()["name"] == "Test Key"
    assert response.json()["key"] == "test-key-123"

def test_list_api_keys(override_auth):
    response = client.get("/api/v1/auth/api-keys")
    assert response.status_code == 200
    assert "keys" in response.json()
    assert response.json()["user_id"] == "test_user"

def test_delete_api_key(override_auth):
    response = client.delete("/api/v1/auth/api-keys/key_123")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

def test_get_current_user_info(override_auth):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 200
    assert response.json()["user_id"] == "test_user"
    assert response.json()["email"] == "test@example.com"
