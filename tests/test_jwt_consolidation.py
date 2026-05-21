import os
import importlib


def test_jwt_shared_module_consistency(monkeypatch):
    # Set required env vars before importing API settings
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    monkeypatch.setenv("APP_ALLOWED_HOSTS", "[\"localhost\"]")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-change-me-please")

    # Import modules after env setup
    jwt_auth = importlib.import_module("api.jwt_auth")
    api_auth = importlib.import_module("api.auth")
    root_auth = importlib.import_module("auth")

    data = {"user_id": 123, "email": "user@example.com"}

    token = jwt_auth.create_access_token(data)
    assert token, "create_access_token returned empty token"

    # verify via shared module
    payload_a = jwt_auth.verify_token(token)
    assert payload_a is not None
    assert payload_a.get("sub") == str(123)

    # verify via API facade
    payload_b = api_auth.verify_token(token)
    assert payload_b is not None
    assert payload_b.get("sub") == str(123)

    # root auth's verify_jwt_token delegates to api.auth, ensure it returns payload
    payload_c = root_auth.verify_jwt_token(token)
    assert payload_c is not None
    assert payload_c.get("sub") == str(123)
