import os

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from core.timeline_payloads import TimelineEventPayload

# api.main -> api.config loads settings at import-time. Ensure required env vars exist.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("JWT_SECRET_KEY", "test-secure-jwt-secret-key-please-change")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

from fastapi import FastAPI, WebSocket, Query
from api.auth import AuthError, TokenExpiredError, InvalidTokenError, verify_token as _verify_token
from services.timeline_realtime import timeline_realtime_bus


from api.websockets.case_timeline import register_case_timeline_endpoint

def _make_test_app():
    app = FastAPI()
    register_case_timeline_endpoint(app)
    return app


@pytest.fixture
def client():
    app = _make_test_app()
    return TestClient(app)


@patch("api.websockets.case_timeline._verify_token")
def test_case_timeline_ws_requires_token(mock_verify_token, client):
    case_id = 123
    try:
        with client.websocket_connect(f"/ws/cases/{case_id}/timeline") as websocket:
            websocket.receive_json()
        pytest.fail("Should have rejected websocket connection without token")
    except Exception as e:
        # fastapi TestClient raises on close/rejection
        assert hasattr(e, "code") or "Authentication required" in str(e) or type(e).__name__ == "WebSocketDisconnect"


@patch("api.websockets.case_timeline._verify_token")
def test_case_timeline_ws_subscribed_and_forwards_event(mock_verify_token, client):
    """
    Validates:
    - auth passes
    - client receives subscribed message
    - timeline bus publish is forwarded to connected websocket
    """
    mock_verify_token.return_value = {"sub": "1"}

    case_id = 1

    # Accept the websocket connection. Provide auth via Sec-WebSocket-Protocol subprotocol
    websocket = client.websocket_connect(
        f"/ws/cases/{case_id}/timeline",
        subprotocols=["access_token", "valid_token"],
    ).__enter__()
    try:
        first = websocket.receive_json()
        assert first["type"] == "subscribed"
        assert first["case_id"] == case_id

        # Publish directly into the realtime bus. This avoids DB/session coupling in the websocket test.
        from services.timeline_realtime import timeline_realtime_bus
        timeline_payload = {
            "type": "timeline_event",
            "case_id": case_id,
            "event_type": "deadline_created",
            "description": "Manual deadline added",
            "timestamp": "2023-01-01T00:00:00+00:00",
            "metadata": {"deadline_id": 999},
            "event_id": 555,
        }

        # timeline_realtime_bus.publish() is async; TestClient is sync,
        # so we run a short event-loop to publish.
        import asyncio

        asyncio.run(timeline_realtime_bus.publish(case_id=case_id, payload=timeline_payload))

        msg = websocket.receive_json()
        validated = TimelineEventPayload.model_validate(msg)
        assert validated.type == "timeline_event"
        assert validated.case_id == case_id
        assert validated.event_type == "deadline_created"
        assert validated.description == "Manual deadline added"
        assert validated.timestamp.isoformat() == "2023-01-01T00:00:00+00:00"
        assert validated.metadata["deadline_id"] == 999
        assert validated.event_id == 555
    finally:
        websocket.close()
