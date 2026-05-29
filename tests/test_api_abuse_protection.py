"""Regression tests for API abuse protection."""

from __future__ import annotations

import os
import json
import math

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-abuse-protection")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost,127.0.0.1")

import pytest
from fastapi import status
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request
from fastapi.testclient import TestClient

from api import limiter as limiter_module
from api.limiter import get_rate_limit_policy, resolve_rate_limit_identifier
from api.validation import ValidationConfig

try:
    from api.middlewares.rate_limit import rate_limit_middleware
except Exception:  # pragma: no cover - optional import in minimal test environments
    rate_limit_middleware = None

try:
    from api.middlewares.request_size import request_size_limit_middleware
except Exception:  # pragma: no cover - optional import in minimal test environments
    request_size_limit_middleware = None


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


class FakeLimiterRedis:
    def __init__(self):
        self.hashes: dict[str, dict[str, object]] = {}
        self.values: dict[str, object] = {}
        self.expiries: dict[str, int] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def pttl(self, key):
        return int(self.expiries.get(key, -2))

    async def hmget(self, key, *fields):
        payload = self.hashes.get(key, {})
        return [payload.get(field) for field in fields]

    async def set(self, key, value, px=None, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        if px is not None:
            self.expiries[key] = int(px)
        elif ex is not None:
            self.expiries[key] = int(ex) * 1000
        return True

    async def incr(self, key):
        current = int(self.values.get(key, 0)) + 1
        self.values[key] = current
        return current

    async def expire(self, key, ttl):
        self.expiries[key] = int(ttl) * 1000
        return True

    async def zincrby(self, key, amount, member):
        bucket = self.zsets.setdefault(key, {})
        bucket[member] = float(bucket.get(member, 0.0)) + float(amount)
        return bucket[member]

    async def zrevrange(self, key, start, end, withscores=False):
        bucket = self.zsets.get(key, {})
        items = sorted(bucket.items(), key=lambda item: item[1], reverse=True)[start : end + 1]
        if withscores:
            return items
        return [member for member, _score in items]


@pytest.mark.asyncio
async def test_rate_limit_middleware_returns_structured_429(monkeypatch):
    if rate_limit_middleware is None:
        pytest.skip("rate_limit middleware unavailable in this environment")
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
async def test_rate_limit_middleware_returns_structured_429_for_reports(monkeypatch):
    if rate_limit_middleware is None:
        pytest.skip("rate_limit middleware unavailable in this environment")
    request = make_request(path="/api/v1/reports/generate", method="POST")

    async def call_next(_request):
        return JSONResponse({"ok": True})

    monkeypatch.setattr("api.middleware.settings.RATE_LIMIT_ENABLED", True)

    async def deny(*args, **kwargs):
        return False

    monkeypatch.setattr("api.middleware.limiter.check_rate_limit", deny)

    async def fake_remaining_ttl(*args, **kwargs):
        return 9

    monkeypatch.setattr("api.middleware.limiter.get_remaining_ttl", fake_remaining_ttl)

    response = await rate_limit_middleware(request, call_next)

    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    payload = json.loads(response.body)
    assert payload["error_code"] == "RATE_LIMIT_EXCEEDED"
    assert payload["retry_after"] == 9


@pytest.mark.asyncio
async def test_rate_limit_middleware_marks_endpoint_overrides(monkeypatch):
    if rate_limit_middleware is None:
        pytest.skip("rate_limit middleware unavailable in this environment")
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
    if request_size_limit_middleware is None:
        pytest.skip("request size middleware unavailable in this environment")
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
    if request_size_limit_middleware is None:
        pytest.skip("request size middleware unavailable in this environment")
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


def test_resolve_rate_limit_identifier_prefers_api_key_identity():
    request = make_request(headers={"X-API-Key": "key_abcdef1234567890.secret-value"}, client_host="10.0.0.1")

    identifier = resolve_rate_limit_identifier(request)

    assert identifier.startswith("api_key:")
    assert len(identifier) > len("api_key:")


@pytest.mark.asyncio
async def test_rate_limiter_smooths_bursts_and_blocks_abuse(monkeypatch):
    fake = FakeLimiterRedis()
    limiter = limiter_module.limiter
    limiter._redis = fake
    limiter.default_burst = 4
    limiter.abuse_threshold = 2
    limiter.abuse_block_seconds = 120

    monkeypatch.setattr(limiter_module.time, "time", lambda: 1_700_000_000.0)

    async def fake_script(keys=None, args=None):
        bucket_key, abuse_key, block_key, stats_key = keys
        now_ms, refill_rate, capacity, cost, abuse_window_ms, abuse_threshold, block_ms, identifier = args

        if fake.expiries.get(block_key, 0) > 0:
            remaining = fake.hashes.get(bucket_key, {}).get("tokens", capacity)
            return [0, fake.expiries[block_key], int(float(remaining)), 0, 1, fake.expiries[block_key]]

        state = fake.hashes.setdefault(bucket_key, {})
        tokens = float(state.get("tokens", capacity))
        ts = float(state.get("ts", now_ms))
        delta = max(0.0, float(now_ms) - ts)
        tokens = min(float(capacity), tokens + (delta * float(refill_rate)))

        allowed = 0
        retry_after = 0
        if tokens >= float(cost):
            tokens -= float(cost)
            allowed = 1
        else:
            retry_after = int(math.ceil((float(cost) - tokens) / float(refill_rate)))

        state["tokens"] = tokens
        state["ts"] = float(now_ms)
        fake.expiries[bucket_key] = max(int(abuse_window_ms), int(math.ceil((float(capacity) / float(refill_rate)) + 1000)))

        violations = int(fake.values.get(abuse_key, 0))
        blocked = 0
        if allowed == 0:
            violations = await fake.incr(abuse_key)
            await fake.expire(abuse_key, int(abuse_window_ms / 1000))
            await fake.zincrby(stats_key, 1, identifier)
            fake.expiries[stats_key] = max(int(abuse_window_ms * 10), int(block_ms))
            if violations >= int(abuse_threshold):
                fake.expiries[block_key] = int(block_ms)
                blocked = 1

        return [allowed, retry_after, int(tokens), violations, blocked, int(fake.expiries.get(block_key, 0))]

    monkeypatch.setattr(limiter, "_script", fake_script)

    endpoint = "POST /api/v1/reports/generate"
    allowed_results = []
    for _ in range(4):
        allowed_results.append(
            await limiter.check_rate_limit(
                identifier="api_key:burst",
                endpoint=endpoint,
                limit=2,
                window_seconds=10,
                request_id="req-burst",
            )
        )

    first_denial = await limiter.check_rate_limit(
        identifier="api_key:burst",
        endpoint=endpoint,
        limit=2,
        window_seconds=10,
        request_id="req-block",
    )

    second_denial = await limiter.check_rate_limit(
        identifier="api_key:burst",
        endpoint=endpoint,
        limit=2,
        window_seconds=10,
        request_id="req-block-2",
    )

    assert allowed_results == [True, True, True, True]
    assert first_denial is False
    assert second_denial is False
    assert await limiter.is_blocked("api_key:burst", endpoint) is True
    report = await limiter.get_abuse_report(endpoint, limit=5)
    assert report[0]["identifier"] == "api_key:burst"
    assert report[0]["events"] >= 1


def test_get_rate_limit_policy_overrides_sensitive_routes():
    auth_rule, matched = get_rate_limit_policy("/api/v1/auth/token", "POST")
    upload_rule, upload_matched = get_rate_limit_policy("/api/v1/analyze/upload", "POST")
    search_rule, search_matched = get_rate_limit_policy("/api/cases/search/text", "GET")
    reports_rule, reports_matched = get_rate_limit_policy("/api/v1/reports/generate", "POST")

    assert matched is True
    assert upload_matched is True
    assert search_matched is True
    assert reports_matched is True
    assert auth_rule.requests == 5
    assert upload_rule.requests == 5
    assert search_rule.requests == 30
    assert reports_rule.requests == 5
