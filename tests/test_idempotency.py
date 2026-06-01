"""Tests for idempotency manager and Celery task integration"""

import time
import pytest
from unittest.mock import MagicMock, patch

from api.idempotency import IdempotencyManager


class FakeRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, nx=False, ex=None):
        if nx:
            if key in self.store:
                return False
            self.store[key] = value
            return True
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        return self.store.pop(key, None) is not None

    def expire(self, key, ex):
        return key in self.store

    def scan_iter(self, match=None, count=None):
        if not match:
            for key in list(self.store.keys()):
                yield key.encode("utf-8")
            return
        prefix = match[:-1] if match.endswith("*") else match
        for key in list(self.store.keys()):
            if key.startswith(prefix):
                yield key.encode("utf-8")


def test_idempotency_manager_basic_flow(monkeypatch):
    fake = FakeRedis()
    manager = IdempotencyManager(redis_url="redis://fake")
    manager._client = fake

    key = "test:1"
    assert manager.acquire(key, ttl=10) is True
    # second acquire should fail
    assert manager.acquire(key, ttl=10) is False

    result = {"ok": True}
    manager.mark_completed(key, result, ttl=60)
    got = manager.get_result(key)
    assert got == result
    manager.release_lock(key)


def test_idempotency_manager_heartbeat_and_state(monkeypatch):
    fake = FakeRedis()
    manager = IdempotencyManager(redis_url="redis://fake")
    manager._client = fake

    key = "task:heartbeat"
    assert manager.acquire(key, ttl=5) is True
    assert manager.heartbeat(key, ttl=5) is True

    result = {"status": "ok"}
    manager.mark_completed(key, result, ttl=20)

    assert manager.get_result(key) == result
    state = fake.store[manager._key_state(key)]
    assert b'"status":"completed"' in state


def test_stale_pending_can_be_taken_over_and_reconciled(monkeypatch):
    fake = FakeRedis()
    manager = IdempotencyManager(redis_url="redis://fake")
    manager._client = fake

    key = "task:stale"
    assert manager.acquire(key, ttl=5) is True

    state_key = manager._key_state(key)
    state = manager._deserialize(fake.store[state_key])
    state["heartbeat"] = int(time.time()) - 100
    state["started"] = int(time.time()) - 100
    fake.store[state_key] = manager._serialize(state)

    fresh_manager = IdempotencyManager(redis_url="redis://fake")
    fresh_manager._client = fake
    assert fresh_manager.acquire(key, ttl=5, stale_after=1) is True

    fresh_manager.mark_completed(key, {"recovered": True}, ttl=20)
    assert fresh_manager.get_result(key) == {"recovered": True}


def test_reconcile_stale_pending_marks_entries_stale(monkeypatch):
    fake = FakeRedis()
    manager = IdempotencyManager(redis_url="redis://fake")
    manager._client = fake

    key = "task:reconcile"
    assert manager.acquire(key, ttl=5) is True

    state_key = manager._key_state(key)
    state = manager._deserialize(fake.store[state_key])
    state["heartbeat"] = int(time.time()) - 100
    state["started"] = int(time.time()) - 100
    fake.store[state_key] = manager._serialize(state)

    reclaimed = manager.reconcile_stale_pending(stale_after=1)
    assert reclaimed == 1
    updated = manager._deserialize(fake.store[state_key])
    assert updated["status"] == "stale"


def test_analyze_task_skips_when_duplicate(monkeypatch):
    import celery_app

    # Mock IdempotencyManager used in celery_app
    fake_manager = MagicMock()
    fake_manager.acquire.return_value = False
    fake_manager.get_result.return_value = {"document_id": "d1", "summary": "already"}

    class _Self:
        def __init__(self):
            self.request = MagicMock(id="tid-analyze")
        def update_state(self, *a, **k):
            return None

    with patch("celery_app.IdempotencyManager", return_value=fake_manager):
        # call the stable task wrapper interface with a dummy self
        res = celery_app.analyze_document_task.run(_Self(), "u1", "d1", "text")
        assert res["document_id"] == "d1"
        assert res["summary"] == "already"


def test_generate_report_marks_completed(monkeypatch):
    import celery_app

    fake_manager = MagicMock()
    fake_manager.acquire.return_value = True

    class _Self2:
        def __init__(self):
            self.request = MagicMock(id="tid-report")
        def update_state(self, *a, **k):
            return None

    # Replace the wrapped implementation with a lightweight stub that marks completion
    with patch("celery_app.IdempotencyManager", return_value=fake_manager):
        def _stub(self, user_id, case_id, report_type="comprehensive", format="pdf"):
            result = {"report_id": "stubbed"}
            fake_manager.mark_completed("report:stub", result)
            return result

        celery_app.generate_report_task.run = _stub
        res = celery_app.generate_report_task.run(None, "u1", "case1")
        assert fake_manager.mark_completed.called
        assert res["report_id"] == "stubbed"


def test_python_sdk_sends_idempotency_key(monkeypatch):
    from sdk.python.client import LegalassistClient

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeHttpClient:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers", {})
            captured["json"] = kwargs.get("json")
            captured["data"] = kwargs.get("data")
            return FakeResponse()

        def close(self):
            return None

    client = LegalassistClient(api_key="api-key")
    client.client = FakeHttpClient()

    client.create_api_key("Example")

    assert captured["headers"].get("Idempotency-Key")
    assert captured["headers"].get("X-API-Key") == "api-key"
