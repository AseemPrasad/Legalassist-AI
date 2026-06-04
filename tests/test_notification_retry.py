import pytest
from unittest.mock import MagicMock, patch
from notification_service import SMSClient, EmailClient
import tenacity


def test_sms_client_retry_on_timeout():
    client = SMSClient()
    # Mock Twilio client
    mock_twilio = MagicMock()
    client.client = mock_twilio
    
    # Configure the create call to raise a timeout exception twice, then succeed
    mock_twilio.messages.create.side_effect = [
        Exception("Gateway Timeout error 504"),
        Exception("Connection reset by peer"),
        MagicMock(sid="successful_retry_sid")
    ]
    
    # Temporarily speed up retries for the test to avoid waiting minutes
    with patch("tenacity.nap.time.sleep", return_value=None):
        success, message_id, error = client.send_sms("+1234567890", "Test message")
        
        assert success is True
        assert message_id == "successful_retry_sid"
        assert error is None
        assert mock_twilio.messages.create.call_count == 3


def test_email_client_retry_on_503():
    client = EmailClient()
    mock_sg = MagicMock()
    client.client = mock_sg
    
    # Configure send call to fail with 503, then succeed
    mock_sg.send.side_effect = [
        Exception("HTTP Error 503: Service Unavailable"),
        MagicMock(status_code=202, headers={"X-Message-ID": "email_success_id"})
    ]
    
    with patch("tenacity.nap.time.sleep", return_value=None):
        success, message_id, error = client.send_email("test@example.com", "Subject", "<p>Content</p>")
        
        assert success is True
        assert message_id == "email_success_id"
        assert error is None
        assert mock_sg.send.call_count == 2
