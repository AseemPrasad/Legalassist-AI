import os
import pytest

# Configure environment variables before importing any config-dependent code
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-that-is-long-enough"

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.csrf import (
    CSRFProtectionMiddleware,
    CSRF_COOKIE_NAME,
    CSRF_TOKEN_HEADER,
    CSRFError,
    validate_csrf_request,
)

# Create a clean, isolated FastAPI app for testing CSRF middleware
app = FastAPI()
app.add_middleware(
    CSRFProtectionMiddleware,
    allowed_hosts={"testserver", "localhost"},
)

@app.get("/test-get")
def handle_get():
    return {"message": "success"}

@app.post("/test-post")
def handle_post():
    return {"message": "success"}

client = TestClient(app)

def test_get_sets_csrf_cookie():
    """Test that a GET request (safe method) sets the csrf_token cookie on the response if missing."""
    client.cookies.clear()
    response = client.get("/test-get")
    assert response.status_code == 200
    assert CSRF_COOKIE_NAME in response.cookies
    token = response.cookies[CSRF_COOKIE_NAME]
    assert len(token) > 0

def test_post_without_origin_bypasses_csrf():
    """Test that a state-mutating request without an Origin header bypasses CSRF checks (for programmatic clients)."""
    client.cookies.clear()
    response = client.post("/test-post")
    assert response.status_code == 200
    assert response.json() == {"message": "success"}

def test_post_with_origin_but_no_cookie_or_header_fails():
    """Test that a state-mutating request with an Origin header but no CSRF cookie/header fails with 403."""
    client.cookies.clear()
    response = client.post("/test-post", headers={"Origin": "http://localhost"})
    assert response.status_code == 403
    payload = response.json()
    assert payload["error_code"] == "CSRF_MISSING_TOKEN"
    assert "Missing CSRF token" in payload["message"]
    assert payload["request_id"]

def test_post_with_origin_and_header_but_no_cookie_fails():
    """Test that a request with header but no cookie fails."""
    client.cookies.clear()
    headers = {
        "Origin": "http://localhost",
        "X-CSRF-Token": "some-token",
    }
    response = client.post("/test-post", headers=headers)
    assert response.status_code == 403
    payload = response.json()
    assert payload["error_code"] == "CSRF_MISSING_COOKIE"
    assert "Missing CSRF cookie" in payload["message"]
    assert payload["request_id"]

def test_post_with_origin_and_mismatched_tokens_fails():
    """Test that mismatched cookie and header values fail."""
    client.cookies.clear()
    headers = {
        "Origin": "http://localhost",
        "X-CSRF-Token": "header-token",
    }
    client.cookies.set(CSRF_COOKIE_NAME, "different-cookie-token")
    response = client.post("/test-post", headers=headers)
    assert response.status_code == 403
    payload = response.json()
    assert payload["error_code"] == "CSRF_TOKEN_MISMATCH"
    assert "CSRF token mismatch" in payload["message"]
    assert payload["request_id"]
    client.cookies.clear()

def test_post_with_matching_tokens_succeeds():
    """Test that matching cookie and header values succeed."""
    client.cookies.clear()
    token = "valid-csrf-token"
    headers = {
        "Origin": "http://localhost",
        "X-CSRF-Token": token,
    }
    client.cookies.set(CSRF_COOKIE_NAME, token)
    response = client.post("/test-post", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": "success"}
    client.cookies.clear()
