import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock, patch

import api.routes.twilio_webhooks as twilio_route
from database import Base, User, CaseDeadline, NotificationLog, NotificationChannel, NotificationStatus, CaseRecord

@pytest.fixture()
def test_db_env():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        bind=engine,
    )
    return TestingSessionLocal

@pytest.fixture()
def client(test_db_env, monkeypatch):
    app = FastAPI()
    app.include_router(twilio_route.router)
    monkeypatch.setattr(twilio_route, "SessionLocal", test_db_env)
    
    # By default, mock validator available
    monkeypatch.setattr(twilio_route, "_VALIDATOR_AVAILABLE", True)
    
    # Mock Twilio auth token
    monkeypatch.setattr("config.Config.get_twilio_auth_token", lambda: "mock_auth_token")
    
    yield TestClient(app)

def test_twilio_webhook_missing_dependency(client, monkeypatch):
    # Mock validator not available
    monkeypatch.setattr(twilio_route, "_VALIDATOR_AVAILABLE", False)
    
    response = client.post(
        "/api/v1/webhooks/twilio/sms-status",
        data={"MessageSid": "SM123", "MessageStatus": "delivered"},
    )
    assert response.status_code == 200
    assert "Response" in response.text

def test_twilio_webhook_missing_auth_token(client, monkeypatch):
    monkeypatch.setattr("config.Config.get_twilio_auth_token", lambda: "")
    
    response = client.post(
        "/api/v1/webhooks/twilio/sms-status",
        data={"MessageSid": "SM123", "MessageStatus": "delivered"},
    )
    assert response.status_code == 200
    assert "Response" in response.text

@patch("api.routes.twilio_webhooks._TwilioRequestValidator")
def test_twilio_webhook_invalid_signature(mock_validator_cls, client, monkeypatch):
    mock_validator = MagicMock()
    mock_validator.validate.return_value = False
    mock_validator_cls.return_value = mock_validator
    
    response = client.post(
        "/api/v1/webhooks/twilio/sms-status",
        headers={"X-Twilio-Signature": "invalid_sig"},
        data={"MessageSid": "SM123", "MessageStatus": "delivered"},
    )
    assert response.status_code == 403

@patch("api.routes.twilio_webhooks._TwilioRequestValidator")
def test_twilio_webhook_validation_exception(mock_validator_cls, client):
    mock_validator = MagicMock()
    mock_validator.validate.side_effect = Exception("Validation exploded")
    mock_validator_cls.return_value = mock_validator
    
    response = client.post(
        "/api/v1/webhooks/twilio/sms-status",
        headers={"X-Twilio-Signature": "valid_sig"},
        data={"MessageSid": "SM123", "MessageStatus": "delivered"},
    )
    assert response.status_code == 403

@patch("api.routes.twilio_webhooks._TwilioRequestValidator")
def test_twilio_webhook_valid_signature_no_log_entry(mock_validator_cls, client):
    mock_validator = MagicMock()
    mock_validator.validate.return_value = True
    mock_validator_cls.return_value = mock_validator
    
    response = client.post(
        "/api/v1/webhooks/twilio/sms-status",
        headers={"X-Twilio-Signature": "valid_sig"},
        data={"MessageSid": "SM123", "MessageStatus": "delivered"},
    )
    assert response.status_code == 200
    assert "Response" in response.text

@patch("api.routes.twilio_webhooks._TwilioRequestValidator")
def test_twilio_webhook_valid_signature_with_log_entry(mock_validator_cls, client, test_db_env):
    mock_validator = MagicMock()
    mock_validator.validate.return_value = True
    mock_validator_cls.return_value = mock_validator
    
    # Seed DB with User, Deadline, and NotificationLog
    db = test_db_env()
    try:
        user = User(email="test@example.com", is_verified=True)
        db.add(user)
        db.commit()
        db.refresh(user)
        
        case = CaseRecord(
            hashed_case_id="case-123",
            case_type="civil",
            jurisdiction="Delhi",
            court_name="High Court",
            judge_name="Judge Alpha",
            plaintiff_type="individual",
            defendant_type="company",
            case_value="1-5L",
            outcome="pending",
            judgment_summary="Sample",
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        
        import datetime
        deadline = CaseDeadline(
            case_id=case.id,
            user_id=user.id,
            case_title="Sample Title",
            deadline_date=datetime.datetime.now(datetime.timezone.utc),
            deadline_type="filing",
            description="Filing",
        )
        db.add(deadline)
        db.commit()
        db.refresh(deadline)
        
        log_entry = NotificationLog(
            deadline_id=deadline.id,
            user_id=user.id,
            channel=NotificationChannel.SMS,
            status=NotificationStatus.PENDING,
            recipient="+911234567890",
            days_before=10,
            message_id="SM123",
        )
        db.add(log_entry)
        db.commit()
    finally:
        db.close()
        
    response = client.post(
        "/api/v1/webhooks/twilio/sms-status",
        headers={"X-Twilio-Signature": "valid_sig"},
        data={"MessageSid": "SM123", "MessageStatus": "delivered"},
    )
    assert response.status_code == 200
    
    # Assert database updated
    db = test_db_env()
    try:
        log = db.query(NotificationLog).filter(NotificationLog.message_id == "SM123").first()
        assert log is not None
        assert log.status == NotificationStatus.SENT
        assert log.delivered_at is not None
    finally:
        db.close()
