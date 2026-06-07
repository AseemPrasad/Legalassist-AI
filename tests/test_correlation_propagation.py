import sys
import types
from types import SimpleNamespace


database_module = types.ModuleType("database")


class _StubDb:
    def close(self):
        return None


def _stub(*args, **kwargs):
    return None


database_module.Attachment = object
database_module.SessionLocal = lambda: _StubDb()
database_module.get_case_by_id = _stub
database_module.get_case_document_by_id = _stub
database_module.update_case_document = _stub
database_module.create_timeline_event = _stub
database_module.cleanup_expired_revoked_tokens = _stub
sys.modules.setdefault("database", database_module)

from celery_app import (
    ContextTask,
    build_task_context_headers,
    enqueue_task_from_http_request,
)


def test_build_task_context_headers_uses_request_id_and_user_id():
    headers = build_task_context_headers(
        request_id="req-123",
        context_user_id="user-7",
        trace_headers={"traceparent": "00-abc-123-01", "tracestate": "vendor=value"},
    )

    assert headers["x-request-id"] == "req-123"
    assert headers["x-correlation-id"] == "req-123"
    assert headers["x-user-id"] == "user-7"
    assert headers["traceparent"] == "00-abc-123-01"
    assert headers["tracestate"] == "vendor=value"


def test_context_task_extracts_request_context_from_headers():
    task_request = SimpleNamespace(
        headers={
            "x-request-id": "req-abc",
            "x-user-id": "user-xyz",
            "traceparent": "00-abc-123-01",
        },
        root_id="root-fallback",
        id="task-fallback",
    )

    context = ContextTask._extract_task_request_context(task_request)

    assert context["request_id"] == "req-abc"
    assert context["user_id"] == "user-xyz"
    assert context["trace_headers"]["traceparent"] == "00-abc-123-01"


def test_enqueue_task_from_http_request_passes_headers_to_apply_async():
    captured = {}

    class FakeTask:
        def apply_async(self, kwargs, headers):
            captured["kwargs"] = kwargs
            captured["headers"] = headers
            return SimpleNamespace(id="task-1")

    http_request = SimpleNamespace(
        state=SimpleNamespace(
            request_id="req-777",
            user_id="user-state",
            trace_headers={"traceparent": "00-req-777-span-01", "tracestate": "state=1"},
        ),
        headers={"X-Correlation-Id": "req-from-header", "X-User-Id": "user-header"},
    )

    result = enqueue_task_from_http_request(
        FakeTask(),
        http_request,
        context_user_id="user-999",
        user_id="task-user",
        document_id="doc-1",
        text="hello",
    )

    assert result.id == "task-1"
    assert captured["kwargs"]["user_id"] == "task-user"
    assert captured["kwargs"]["document_id"] == "doc-1"
    assert captured["headers"]["x-request-id"] == "req-777"
    assert captured["headers"]["x-correlation-id"] == "req-777"
    assert captured["headers"]["x-user-id"] == "user-999"
    assert captured["headers"]["traceparent"] == "00-req-777-span-01"
    assert captured["headers"]["tracestate"] == "state=1"
