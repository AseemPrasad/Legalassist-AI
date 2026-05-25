from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from twilio.request_validator import RequestValidator

import api.main as api_main
from config import Config
from database import get_db
from db.base import Base
from db.models.auth import User
from db.models.cases import Case, CaseDeadline, CaseStatus
from db.models.notifications import NotificationChannel, NotificationLog, NotificationStatus


SENDGRID_PUBLIC_KEY = "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE83T4O/n84iotIvIW4mdBgQ/7dAfSmpqIM8kF9mN1flpVKS3GRqe62gw+2fNNRaINXvVpiglSI8eNEc6wEA3F+g=="
SENDGRID_SIGNATURE = "MEUCIGHQVtGj+Y3LkG9fLcxf3qfI10QysgDWmMOVmxG0u6ZUAiEAyBiXDWzM+uOe5W0JuG+luQAbPIqHh89M15TluLtEZtM="
SENDGRID_TIMESTAMP = "1600112502"
SENDGRID_PAYLOAD = (
    json.dumps(
        [
            {
                "email": "hello@world.com",
                "event": "dropped",
                "reason": "Bounced Address",
                "sg_event_id": "ZHJvcC0xMDk5NDkxOS1MUnpYbF9OSFN0T0doUTRrb2ZTbV9BLTA",
                "sg_message_id": "LRzXl_NHStOGhQ4kofSm_A.filterdrecv-p3mdw1-756b745b58-kmzbl-18-5F5FC76C-9.0",
                "smtp-id": "<LRzXl_NHStOGhQ4kofSm_A@ismtpd0039p1iad1.sendgrid.net>",
                "timestamp": 1600112492,
            }
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\r\n"
)


@pytest.fixture()
def test_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def client(test_db, monkeypatch):
    api_main.app.dependency_overrides[get_db] = lambda: test_db
    monkeypatch.setattr(Config, "get_twilio_auth_token", classmethod(lambda cls: "twilio-token"))
    monkeypatch.setattr(Config, "get_sendgrid_event_webhook_public_key", classmethod(lambda cls: SENDGRID_PUBLIC_KEY))
    with TestClient(api_main.app) as test_client:
        yield test_client
    api_main.app.dependency_overrides.clear()


def _seed_notification_log(db, message_id: str, channel: NotificationChannel):
    user = User(email="webhook@example.com")
    db.add(user)
    db.commit()
    db.refresh(user)

    case = Case(
        user_id=user.id,
        case_number="CASE-WEBHOOK",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Webhook Case",
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    deadline = CaseDeadline(
        user_id=user.id,
        case_id=case.id,
        case_title="Webhook Case",
        deadline_date=datetime.now(timezone.utc) + timedelta(days=10),
        deadline_type="appeal",
    )
    db.add(deadline)
    db.commit()
    db.refresh(deadline)

    log = NotificationLog(
        deadline_id=deadline.id,
        user_id=user.id,
        channel=channel,
        recipient="recipient@example.com",
        days_before=10,
        status=NotificationStatus.SENT,
        message_id=message_id,
        sent_at=datetime.now(timezone.utc),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def test_twilio_webhook_marks_notification_delivered(client, test_db):
    message_sid = "SM123456789"
    log = _seed_notification_log(test_db, message_sid, NotificationChannel.SMS)

    params = {
        "MessageSid": message_sid,
        "MessageStatus": "delivered",
        "SmsSid": message_sid,
        "SmsStatus": "delivered",
    }
    signature = RequestValidator("twilio-token").compute_signature("http://testserver/api/v1/webhooks/twilio", params)

    response = client.post(
        "/api/v1/webhooks/twilio",
        data=params,
        headers={"X-Twilio-Signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["updated"] is True

    refreshed = test_db.query(NotificationLog).filter(NotificationLog.id == log.id).one()
    assert refreshed.status == NotificationStatus.DELIVERED
    assert refreshed.delivered_at is not None
    assert refreshed.sent_at is not None
    assert refreshed.failed_at is None


def test_sendgrid_webhook_marks_notification_failed(client, test_db):
    message_id = "LRzXl_NHStOGhQ4kofSm_A.filterdrecv-p3mdw1-756b745b58-kmzbl-18-5F5FC76C-9.0"
    log = _seed_notification_log(test_db, message_id, NotificationChannel.EMAIL)

    response = client.post(
        "/api/v1/webhooks/sendgrid",
        content=SENDGRID_PAYLOAD.encode("utf-8"),
        headers={
            "X-Twilio-Email-Event-Webhook-Signature": SENDGRID_SIGNATURE,
            "X-Twilio-Email-Event-Webhook-Timestamp": SENDGRID_TIMESTAMP,
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["events"] == 1
    assert body["updated"] == 1

    refreshed = test_db.query(NotificationLog).filter(NotificationLog.id == log.id).one()
    assert refreshed.status == NotificationStatus.FAILED
    assert refreshed.failed_at is not None
    assert "Bounced Address" in (refreshed.error_message or "")
    assert refreshed.delivered_at is None