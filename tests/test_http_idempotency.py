"""Tests for HTTP idempotency-key behavior."""

import os

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-idempotency")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost,127.0.0.1")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import idempotency_middleware, http_idempotency_manager


class FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        return self.store.pop(key, None) is not None


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
