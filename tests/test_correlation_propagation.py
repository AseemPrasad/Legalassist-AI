from types import SimpleNamespace

from celery_app import (
    ContextTask,
    build_task_context_headers,
    enqueue_task_from_http_request,
)


def test_build_task_context_headers_uses_request_id_and_user_id():
    headers = build_task_context_headers(request_id="req-123", context_user_id="user-7")

    assert headers["x-request-id"] == "req-123"
    assert headers["x-correlation-id"] == "req-123"
    assert headers["x-user-id"] == "user-7"


def test_context_task_extracts_request_context_from_headers():
    task_request = SimpleNamespace(
        headers={
            "x-request-id": "req-abc",
            "x-user-id": "user-xyz",
        },
        root_id="root-fallback",
        id="task-fallback",
    )

    context = ContextTask._extract_task_request_context(task_request)

    assert context["request_id"] == "req-abc"
    assert context["user_id"] == "user-xyz"


def test_enqueue_task_from_http_request_passes_headers_to_apply_async():
    captured = {}

    class FakeTask:
        def apply_async(self, kwargs, headers):
            captured["kwargs"] = kwargs
            captured["headers"] = headers
            return SimpleNamespace(id="task-1")

    http_request = SimpleNamespace(
        state=SimpleNamespace(request_id="req-777", user_id="user-state"),
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
