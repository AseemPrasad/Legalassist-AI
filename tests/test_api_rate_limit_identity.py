import os
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")

from fastapi import HTTPException

from api.limiter import resolve_rate_limit_identifier, limiter


class _FakeRequest:
    def __init__(self, headers=None, client_host="203.0.113.10"):
        self.headers = headers or {}
        self.client = SimpleNamespace(host=client_host)


def test_rate_limit_identifier_uses_verified_auth_token(monkeypatch):
    request = _FakeRequest(
        headers={
            "X-User-Id": "attacker-controlled-id",
            "Authorization": "Bearer signed-user-42-token",
        }
    )

    monkeypatch.setattr(
        "api.limiter.verify_token",
        lambda token: {"sub": "user-42"} if token == "signed-user-42-token" else (_ for _ in ()).throw(HTTPException(status_code=401)),
    )

    assert resolve_rate_limit_identifier(request) == "user:user-42"


def test_rate_limit_identifier_falls_back_to_ip_when_auth_is_missing(monkeypatch):
    request = _FakeRequest(headers={"X-User-Id": "spoofed"}, client_host="198.51.100.7")

    monkeypatch.setattr("api.limiter.verify_token", lambda token: (_ for _ in ()).throw(HTTPException(status_code=401)))

    assert resolve_rate_limit_identifier(request) == "ip:198.51.100.7"


def test_rate_limit_key_is_versioned_and_not_minute_bucketed():
    key = limiter._generate_key("user:42", "GLOBAL_API_LIMIT")

    assert key.startswith("ratelimit:v2:")
    assert "// 60" not in key
