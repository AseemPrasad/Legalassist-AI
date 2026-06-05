import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.timeline_payloads import TimelineEventPayload, TimelineSubscribedPayload
from db.base import Base
from db.models.cases import Case, CaseDeadline, CaseStatus
from db.models.notifications import NotificationChannel, NotificationLog, NotificationStatus
from db.session import get_db

# api.main -> api.config loads settings at import-time. Ensure required env vars exist.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/1")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
os.environ.setdefault("JWT_SECRET_KEY", "test-secure-jwt-secret-key-please-change")
os.environ.setdefault("APP_ALLOWED_HOSTS", "localhost")

from services.timeline_realtime import timeline_realtime_bus
from services.timeline_service import timeline_service


from api.websockets.case_timeline import register_case_timeline_endpoint

def _make_test_app():
    app = FastAPI()
    register_case_timeline_endpoint(app)
    return app


@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    db = SessionLocal()
    yield db
    db.close()


def _seed_case(test_db, user_id: int, case_number: str) -> Case:
    created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    case = Case(
        user_id=user_id,
        case_number=case_number,
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title=f"Case {case_number}",
        created_at=created_at,
        updated_at=created_at + timedelta(days=1),
    )
    test_db.add(case)
    test_db.commit()
    test_db.refresh(case)
    return case


@pytest.fixture
def client(test_db):
    app = _make_test_app()
    app.dependency_overrides[get_db] = lambda: test_db
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
def test_case_timeline_ws_subscribed_and_forwards_event(mock_verify_token, client, test_db):
    """
    Validates:
    - auth passes
    - client receives subscribed message
    - timeline bus publish is forwarded to connected websocket
    """
    mock_verify_token.return_value = {"sub": "1"}

    case = _seed_case(test_db, user_id=1, case_number="2023-CV-00001")
    case_id = case.id

    # Accept the websocket connection. Provide auth via Sec-WebSocket-Protocol subprotocol
    websocket = client.websocket_connect(
        f"/ws/cases/{case_id}/timeline",
        subprotocols=["access_token", "valid_token"],
    ).__enter__()
    try:
        first = websocket.receive_json()
        subscribed = TimelineSubscribedPayload.model_validate(first)
        assert subscribed.schema_version == TimelineEventPayload.CURRENT_SCHEMA_VERSION
        assert subscribed.type == "subscribed"
        assert subscribed.case_id == case_id

        # Publish directly into the realtime bus. This avoids DB/session coupling in the websocket test.
        from services.timeline_realtime import timeline_realtime_bus
        timeline_payload = TimelineEventPayload(
            schema_version=TimelineEventPayload.CURRENT_SCHEMA_VERSION,
            type="timeline_event",
            case_id=case_id,
            event_type="deadline_created",
            description="Manual deadline added",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            metadata={"deadline_id": 999},
            event_id=555,
        )

        # timeline_realtime_bus.publish() is async; TestClient is sync,
        # so we run a short event-loop to publish.
        import asyncio

        asyncio.run(timeline_realtime_bus.publish(case_id=case_id, payload=timeline_payload.model_dump(mode="json")))

        msg = websocket.receive_json()
        validated = TimelineEventPayload.model_validate(msg)
        assert set(validated.model_dump(mode="json")) == {
            "schema_version",
            "type",
            "case_id",
            "event_type",
            "description",
            "timestamp",
            "metadata",
            "event_id",
        }
        assert validated.schema_version == TimelineEventPayload.CURRENT_SCHEMA_VERSION
        assert validated.type == "timeline_event"
        assert validated.case_id == case_id
        assert validated.event_type == "deadline_created"
        assert validated.description == "Manual deadline added"
        assert validated.timestamp.isoformat() == "2023-01-01T00:00:00+00:00"
        assert validated.metadata["deadline_id"] == 999
        assert validated.event_id == 555
    finally:
        websocket.close()


@patch("api.websockets.case_timeline._verify_token")
def test_case_timeline_ws_forwards_notification_event(mock_verify_token, client, test_db):
    mock_verify_token.return_value = {"sub": "1"}

    case = _seed_case(test_db, user_id=1, case_number="2023-CV-00010")
    deadline = CaseDeadline(
        user_id=1,
        case_id=case.id,
        case_title=case.title or f"Case {case.case_number}",
        deadline_date=datetime(2025, 2, 1, tzinfo=timezone.utc),
        deadline_type="appeal",
    )
    test_db.add(deadline)
    test_db.commit()
    test_db.refresh(deadline)

    notification_log = NotificationLog(
        deadline_id=deadline.id,
        user_id=1,
        channel=NotificationChannel.SMS,
        recipient="+911234567890",
        days_before=7,
        status=NotificationStatus.SENT,
        message_id="SM-notification-1",
    )
    test_db.add(notification_log)
    test_db.commit()
    test_db.refresh(notification_log)

    websocket = client.websocket_connect(
        f"/ws/cases/{case.id}/timeline",
        subprotocols=["access_token", "valid_token"],
    ).__enter__()
    try:
        subscribed = TimelineSubscribedPayload.model_validate(websocket.receive_json())
        assert subscribed.case_id == case.id

        timeline_service.record_notification_event(
            db=test_db,
            notification_log=notification_log,
            status=NotificationStatus.DELIVERED,
            provider="twilio",
            metadata={"message_id": "SM-notification-1"},
        )

        msg = websocket.receive_json()
        validated = TimelineEventPayload.model_validate(msg)
        assert validated.event_type == "notification_delivered"
        assert validated.case_id == case.id
        assert validated.metadata["notification_log_id"] == notification_log.id
        assert validated.metadata["status"] == "delivered"
        assert validated.metadata["provider"] == "twilio"
        assert validated.metadata["channel"] == "sms"
    finally:
        websocket.close()


@patch("api.websockets.case_timeline._verify_token")
def test_case_timeline_ws_rejects_other_users_case(mock_verify_token, client, test_db):
    mock_verify_token.return_value = {"sub": "1"}
    case = _seed_case(test_db, user_id=99, case_number="2023-CV-00099")

    with pytest.raises(Exception) as exc_info:
        with client.websocket_connect(
            f"/ws/cases/{case.id}/timeline",
            subprotocols=["access_token", "valid_token"],
        ) as websocket:
            websocket.receive_json()

    assert "Forbidden" in str(exc_info.value) or getattr(exc_info.value, "code", None) in {1008, 4003}


@patch("api.websockets.case_timeline._verify_token")
def test_case_timeline_ws_isolates_users_by_case_room(mock_verify_token, client, test_db):
    def fake_verify_token(token):
        if token == "user-1-token":
            return {"sub": "1"}
        if token == "user-2-token":
            return {"sub": "2"}
        raise AssertionError(f"Unexpected token: {token}")

    mock_verify_token.side_effect = fake_verify_token

    case_a = _seed_case(test_db, user_id=1, case_number="2023-CV-00001")
    case_b = _seed_case(test_db, user_id=2, case_number="2023-CV-00002")

    websocket_a = client.websocket_connect(
        f"/ws/cases/{case_a.id}/timeline",
        subprotocols=["access_token", "user-1-token"],
    ).__enter__()
    websocket_b = client.websocket_connect(
        f"/ws/cases/{case_b.id}/timeline",
        subprotocols=["access_token", "user-2-token"],
    ).__enter__()

    try:
        assert TimelineSubscribedPayload.model_validate(websocket_a.receive_json()) == TimelineSubscribedPayload(case_id=case_a.id)
        assert TimelineSubscribedPayload.model_validate(websocket_b.receive_json()) == TimelineSubscribedPayload(case_id=case_b.id)

        async def publish(case_id: int, event_id: int, description: str):
            await timeline_realtime_bus.publish(
                case_id=case_id,
                payload=TimelineEventPayload(
                    schema_version=TimelineEventPayload.CURRENT_SCHEMA_VERSION,
                    type="timeline_event",
                    case_id=case_id,
                    event_type="deadline_created",
                    description=description,
                    timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    metadata={"deadline_id": event_id},
                    event_id=event_id,
                ).model_dump(mode="json"),
            )

        import asyncio

        asyncio.run(publish(case_a.id, 101, "Case A update"))
        assert TimelineEventPayload.model_validate(websocket_a.receive_json()).description == "Case A update"

        asyncio.run(publish(case_b.id, 202, "Case B update"))
        assert TimelineEventPayload.model_validate(websocket_b.receive_json()).description == "Case B update"
    finally:
        websocket_a.close()
        websocket_b.close()


@patch("api.websockets.case_timeline._verify_token")
def test_case_timeline_ws_rate_limited(mock_verify_token, client, test_db, monkeypatch):
    mock_verify_token.return_value = {"sub": "1"}
    case = _seed_case(test_db, user_id=1, case_number="2023-CV-00003")

    async def deny(*args, **kwargs):
        return False

    async def fake_remaining_ttl(*args, **kwargs):
        return 7

    monkeypatch.setattr("api.limiter.limiter.check_rate_limit", deny)
    monkeypatch.setattr("api.limiter.limiter.get_remaining_ttl", fake_remaining_ttl)

    with pytest.raises(Exception) as exc_info:
        with client.websocket_connect(
            f"/ws/cases/{case.id}/timeline",
            subprotocols=["access_token", "valid_token"],
        ) as websocket:
            websocket.receive_json()

    assert hasattr(exc_info.value, "code") or "1013" in str(exc_info.value)
