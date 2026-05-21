"""Regression tests for API abuse protection."""

from __future__ import annotations

import os
import json

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-abuse-protection")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost,127.0.0.1")

import pytest
from fastapi import status
from fastapi.responses import JSONResponse
from starlette.requests import Request

from api import limiter as limiter_module
from api.limiter import get_rate_limit_policy, resolve_rate_limit_identifier
from api.middleware import rate_limit_middleware, request_size_limit_middleware
from api.middlewares import register_middlewares
from api.validation import ValidationConfig


def make_request(path: str = "/api/v1/analyze/document", method: str = "POST", headers: dict | None = None, client_host: str = "127.0.0.1") -> Request:
    raw_headers = [(key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": raw_headers,
        "client": (client_host, 12345),
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_rate_limit_middleware_returns_structured_429(monkeypatch):
    request = make_request(path="/api/v1/analyze/upload", method="POST", headers={"X-Correlation-Id": "req-1"})

    async def call_next(_request):
        return JSONResponse({"ok": True})

    monkeypatch.setattr("api.middleware.settings.RATE_LIMIT_ENABLED", True)
    async def deny(*args, **kwargs):
        return False

    monkeypatch.setattr("api.middleware.limiter.check_rate_limit", deny)

    async def fake_remaining_ttl(*args, **kwargs):
        return 17

    monkeypatch.setattr("api.middleware.limiter.get_remaining_ttl", fake_remaining_ttl)

    response = await rate_limit_middleware(request, call_next)

    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert response.headers["Retry-After"] == "17"
    payload = json.loads(response.body)
    assert payload["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert payload["retry_after"] == 17


@pytest.mark.asyncio
async def test_rate_limit_middleware_marks_endpoint_overrides(monkeypatch):
    request = make_request(path="/api/cases/search/text", method="GET")

    async def call_next(_request):
        return JSONResponse({"ok": True})

    monkeypatch.setattr("api.middleware.settings.RATE_LIMIT_ENABLED", True)

    async def allow(*args, **kwargs):
        return True

    monkeypatch.setattr("api.middleware.limiter.check_rate_limit", allow)

    response = await rate_limit_middleware(request, call_next)

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["X-RateLimit-Scope"] == "endpoint"
    assert response.headers["X-RateLimit-Limit"] == "30"


@pytest.mark.asyncio
async def test_rate_limit_middleware_fails_closed_when_redis_errors(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr("api.limiter.DistributedRateLimiter.get_redis", boom)

    allowed = await limiter_module.limiter.check_rate_limit(
        identifier="ip:203.0.113.10",
        endpoint="POST /api/v1/reports/generate",
        limit=5,
        window_seconds=60,
    )

    assert allowed is False


@pytest.mark.asyncio
async def test_request_size_limit_middleware_rejects_large_json(monkeypatch):
    monkeypatch.setattr(ValidationConfig, "MAX_JSON_BODY", 100)
    monkeypatch.setattr(ValidationConfig, "MAX_UPLOAD_SIZE", 200)

    request = make_request(path="/api/v1/cases", method="POST", headers={"content-length": "150"})

    async def call_next(_request):
        return JSONResponse({"ok": True})

    response = await request_size_limit_middleware(request, call_next)

    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    payload = json.loads(response.body)
    assert payload["error_code"] == "PAYLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_request_size_limit_middleware_allows_upload_threshold(monkeypatch):
    monkeypatch.setattr(ValidationConfig, "MAX_JSON_BODY", 100)
    monkeypatch.setattr(ValidationConfig, "MAX_UPLOAD_SIZE", 200)

    request = make_request(path="/api/v1/analyze/upload", method="POST", headers={"content-length": "150"})

    async def call_next(_request):
        return JSONResponse({"ok": True})

    response = await request_size_limit_middleware(request, call_next)

    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.body) == {"ok": True}


def test_resolve_rate_limit_identifier_prefers_verified_user(monkeypatch):
    request = make_request(headers={"Authorization": "Bearer token"}, client_host="10.0.0.1")

    def fake_verify_token(_token):
        return {"sub": "user-123"}

    monkeypatch.setattr("api.limiter.verify_token", fake_verify_token)

    assert resolve_rate_limit_identifier(request) == "user:user-123"


def test_get_rate_limit_policy_overrides_sensitive_routes():
    auth_rule, matched = get_rate_limit_policy("/api/v1/auth/token", "POST")
    upload_rule, upload_matched = get_rate_limit_policy("/api/v1/analyze/upload", "POST")
    search_rule, search_matched = get_rate_limit_policy("/api/cases/search/text", "GET")

    assert matched is True
    assert upload_matched is True
    assert search_matched is True
    assert auth_rule.requests == 5
    assert upload_rule.requests == 5
    assert search_rule.requests == 30


def test_register_middlewares_preserves_order(monkeypatch):
    class DummyApp:
        def __init__(self):
            self.calls = []

        def middleware(self, _type):
            def decorator(func):
                self.calls.append(func.__name__)
                return func

            return decorator

    monkeypatch.setattr("api.middlewares.rate_limit.settings.RATE_LIMIT_ENABLED", True)

    app = DummyApp()
    register_middlewares(app)

    assert app.calls == [
        "request_size_limit_middleware",
        "idempotency_middleware",
        "add_correlation_id_middleware",
        "logging_middleware",
        "error_handling_middleware",
        "rate_limit_middleware",
    ]
