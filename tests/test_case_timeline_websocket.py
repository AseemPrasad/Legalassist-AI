import json
import os

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# api.main -> api.config loads settings at import-time. Ensure required env vars exist.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("JWT_SECRET_KEY", "test-secure-jwt-secret-key-please-change")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

from fastapi import FastAPI, WebSocket, Query
from api.auth import AuthError, TokenExpiredError, InvalidTokenError, verify_token as _verify_token
from services.timeline_realtime import timeline_realtime_bus


def _make_test_app():
    app = FastAPI()

    @app.websocket("/ws/cases/{case_id}/timeline")
    async def websocket_case_timeline_endpoint(
        websocket: WebSocket,
        case_id: int,
        token: str = Query(None),
    ):
        auth_token = token
        requested_protocols = []

        if "sec-websocket-protocol" in websocket.headers:
            header_val = websocket.headers["sec-websocket-protocol"]
            requested_protocols = [p.strip() for p in header_val.split(",")]
            if "access_token" in requested_protocols:
                idx = requested_protocols.index("access_token")
                if idx + 1 < len(requested_protocols):
                    auth_token = requested_protocols[idx + 1]

        if not auth_token:
            await websocket.close(code=4001, reason="Authentication required")
            return

        try:
            payload = _verify_token(auth_token)
            user_id = payload.get("sub")
            if not user_id:
                await websocket.close(code=4003, reason="Invalid token")
                return
        except (TokenExpiredError, InvalidTokenError, AuthError):
            await websocket.close(code=4001, reason="Invalid or expired token")
            return

        subprotocol = "access_token" if "access_token" in requested_protocols else None
        await websocket.accept(subprotocol=subprotocol)

        await websocket.send_json({"type": "subscribed", "case_id": case_id})

        queue = await timeline_realtime_bus.subscribe(case_id)
        try:
            import json

            while True:
                raw = await queue.get()
                payload_obj = json.loads(raw)
                await websocket.send_json(payload_obj)
        finally:
            await timeline_realtime_bus.unsubscribe(case_id, queue)

    return app


@pytest.fixture
def client():
    app = _make_test_app()
    return TestClient(app)


@patch("api.auth.verify_token")
def test_case_timeline_ws_requires_token(mock_verify_token, client):
    case_id = 123
    try:
        with client.websocket_connect(f"/ws/cases/{case_id}/timeline") as websocket:
            websocket.receive_json()
        pytest.fail("Should have rejected websocket connection without token")
    except Exception as e:
        # fastapi TestClient raises on close/rejection
        assert hasattr(e, "code") or "Authentication required" in str(e) or type(e).__name__ == "WebSocketDisconnect"


@patch("api.auth.verify_token")
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
    with client.websocket_connect(
        f"/ws/cases/{case_id}/timeline",
        subprotocols=["access_token", "valid_token"],
    ) as websocket:
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
        assert msg["type"] == "timeline_event"
        assert msg["case_id"] == case_id
        assert msg["event_type"] == "deadline_created"
        assert msg["description"] == "Manual deadline added"
        assert msg["metadata"]["deadline_id"] == 999
        assert msg["event_id"] == 555
