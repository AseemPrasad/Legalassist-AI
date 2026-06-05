import os
import re
from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest

os.environ["DEBUG"] = "0"
os.environ["TESTING"] = "1"
os.environ["JWT_SECRET"] = "test-secret-key-that-is-long-enough"
os.environ["APP_ALLOWED_HOSTS"] = "localhost"

from database import (
    NotificationChannel,
)
import auth
import scheduler
from notification_service import EmailClient, NotificationResult, SMSClient


def _assert_no_sensitive_patterns(text: str) -> None:
    assert not re.search(r"\b[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b", text)
    assert "123456" not in text
    assert "654321" not in text


def test_auth_logs_redact_jwt_and_email(capfd, monkeypatch):
    monkeypatch.setenv("SENDGRID_API_KEY", "test-key")
    fake_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signaturepart"

    fake_client = Mock()
    fake_client.send.side_effect = Exception(f"provider rejected token {fake_jwt} for user@example.com")
    monkeypatch.setattr(auth, "sendgrid", Mock(SendGridAPIClient=lambda api_key: fake_client))
    monkeypatch.setattr(auth, "Mail", lambda **kwargs: object())

    assert auth.send_otp_email("user@example.com", "123456") is False

    captured = capfd.readouterr()
    output = captured.out + captured.err
    _assert_no_sensitive_patterns(output)
    assert "[redacted-token]" in output
    assert "user@example.com" not in output


def test_notification_logs_redact_recipient_and_body(capfd, monkeypatch):
    monkeypatch.setattr("config.Config.TESTING", True)
    monkeypatch.setattr("config.Config.DEBUG", True)
    monkeypatch.setattr("config.Config.TWILIO_ACCOUNT_SID", "")
    monkeypatch.setattr("config.Config.TWILIO_FROM_NUMBER", "")
    monkeypatch.setattr("config.Config.SENDGRID_FROM_EMAIL", "noreply@example.com")
    monkeypatch.setenv("SENDGRID_API_KEY", "")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "")

    sms_client = SMSClient()
    email_client = EmailClient()

    sms_client.send_sms("+91-9876543210", "Confidential body with OTP 654321")
    email_client.send_email("user@example.com", "Subject containing user@example.com", "<p>body 654321</p>")

    captured = capfd.readouterr()
    output = captured.out + captured.err
    _assert_no_sensitive_patterns(output)
    assert "Confidential body" not in output
    assert "Subject containing" not in output
    assert "u***r@example.com" in output


def test_scheduler_logs_mask_notification_recipients(capfd):
    fake_deadline = SimpleNamespace(
        case_id=1,
        user_id=1,
        days_until_deadline=lambda: 30,
    )
    fake_pref = SimpleNamespace(user_id=1, timezone="UTC")

    with patch("scheduler.SessionLocal", return_value=Mock()), \
            patch("scheduler.init_db", return_value=None), \
         patch("scheduler.get_reminder_dispatch_candidates", return_value=[(fake_deadline, 30, fake_pref)]), \
         patch("scheduler.notification_service.send_reminders") as mock_send_reminders:
        mock_send_reminders.return_value = [
            NotificationResult(success=True, channel=NotificationChannel.SMS, recipient="+91-9876543210", message_id="sms_123", error=None),
            NotificationResult(success=True, channel=NotificationChannel.EMAIL, recipient="user@example.com", message_id="email_123", error=None),
        ]
        scheduler.check_and_send_reminders()

    captured = capfd.readouterr()
    output = captured.out + captured.err
    _assert_no_sensitive_patterns(output)
    assert "user@example.com" not in output
    assert "+91-9876543210" not in output
    assert "u***r@example.com" in output
    assert mock_send_reminders.called
