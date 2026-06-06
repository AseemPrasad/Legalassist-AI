"""Tests for HTTP idempotency-key behavior."""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import types
from unittest.mock import MagicMock

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-idempotency")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost,127.0.0.1")
sys.modules.setdefault("jwt", MagicMock())
passlib_module = types.ModuleType("passlib")
passlib_context_module = types.ModuleType("passlib.context")


class _FakeCryptContext:
    def __init__(self, *args, **kwargs):
        pass

    def verify(self, *args, **kwargs):
        return True

    def hash(self, password):
        return f"hashed:{password}"


passlib_context_module.CryptContext = _FakeCryptContext
passlib_module.context = passlib_context_module
sys.modules.setdefault("passlib", passlib_module)
sys.modules.setdefault("passlib.context", passlib_context_module)

database_module = types.ModuleType("database")


class _FakeDbSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        return None


database_module.SessionLocal = lambda: _FakeDbSession()
database_module.APIKey = object
database_module.User = object
database_module.db_session = lambda: _FakeDbSession()
database_module.is_token_revoked = lambda *args, **kwargs: False
database_module.revoke_token = lambda *args, **kwargs: True
sys.modules.setdefault("database", database_module)

jwt_auth_module = types.ModuleType("api.jwt_auth")
jwt_auth_module.AuthError = Exception
jwt_auth_module.TokenExpiredError = Exception
jwt_auth_module.InvalidTokenError = Exception
jwt_auth_module.create_access_token = lambda *args, **kwargs: "token"
jwt_auth_module.verify_token = lambda *args, **kwargs: None
jwt_auth_module.revoke_jwt_token = lambda *args, **kwargs: True
sys.modules.setdefault("api.jwt_auth", jwt_auth_module)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import idempotency_middleware, http_idempotency_manager


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.lock = threading.Lock()

    def set(self, key, value, nx=False, ex=None):
        with self.lock:
            if nx and key in self.store:
                return False
            self.store[key] = value
            return True

    def get(self, key):
        with self.lock:
            return self.store.get(key)

    def delete(self, key):
        with self.lock:
            return self.store.pop(key, None) is not None


def test_concurrent_duplicate_requests_replay_same_result(monkeypatch):
    app, counter = build_app()
    client = TestClient(app)

    fake = FakeRedis()
    http_idempotency_manager._client = fake

    @app.post("/slow-resources")
    async def create_slow_resource(payload: dict):
        time.sleep(0.2)
        counter["value"] += 1
        return {"created": True, "counter": counter["value"], "payload": payload}

    headers = {"Idempotency-Key": "resource-key-concurrent"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(client.post, "/slow-resources", json={"name": "alpha"}, headers=headers)
        time.sleep(0.05)
        second_future = executor.submit(client.post, "/slow-resources", json={"name": "alpha"}, headers=headers)
        first = first_future.result(timeout=5)
        second = second_future.result(timeout=5)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert second.headers.get("X-Idempotent-Replay") == "true"
    assert counter["value"] == 1


def build_app():
    app = FastAPI()
    app.middleware("http")(idempotency_middleware)

    counter = {"value": 0}

    @app.post("/resources")
    async def create_resource(payload: dict):
        counter["value"] += 1
        return {"created": True, "counter": counter["value"], "payload": payload}

    return app, counter


def test_write_requests_require_idempotency_key():
    app, _counter = build_app()
    client = TestClient(app)

    response = client.post("/resources", json={"name": "alpha"})

    assert response.status_code == 428
    assert response.json()["error_code"] == "IDEMPOTENCY_KEY_REQUIRED"


def test_repeated_write_requests_return_same_result(monkeypatch):
    app, counter = build_app()
    client = TestClient(app)

    fake = FakeRedis()
    http_idempotency_manager._client = fake

    headers = {"Idempotency-Key": "resource-key-1"}

    first = client.post("/resources", json={"name": "alpha"}, headers=headers)
    second = client.post("/resources", json={"name": "alpha"}, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert second.headers.get("X-Idempotent-Replay") == "true"
    assert counter["value"] == 1


def test_same_json_with_different_whitespace_reuses_idempotency_result(monkeypatch):
    app, counter = build_app()
    client = TestClient(app)

    fake = FakeRedis()
    http_idempotency_manager._client = fake

    headers = {
        "Idempotency-Key": "resource-key-2",
        "Content-Type": "application/json",
    }

    first = client.post(
        "/resources",
        content='{"name":"alpha","tags":[1,2]}',
        headers=headers,
    )
    second = client.post(
        "/resources",
        content='''{
            "tags": [1, 2],
            "name": "alpha"
        }''',
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert second.headers.get("X-Idempotent-Replay") == "true"
    assert counter["value"] == 1
