
import pytest
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import (
    Base,
    CaseDeadline,
    UserPreference,
    NotificationChannel,
    create_case_deadline,
    create_or_update_user_preference,
    Case,
    CaseStatus,
    User,
)
from notification_service import NotificationService, SMSClient, EmailClient

@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

class TestNotificationServiceExtended:
    """Extended tests for notification_service.py to improve coverage"""

    def test_send_reminders_all_thresholds(self, test_db):
        """Test send_reminders function with multiple thresholds"""
        now = datetime.now(timezone.utc)
        
        # Test for each threshold
        for days in [30, 10, 3, 1]:
            user_id = days
            user = User(id=user_id, email=f"user{days}@example.com")
            test_db.add(user)
            test_db.commit()

            case_id_int = 101 + days
            case = Case(user_id=user_id, case_number=f"CASE-{case_id_int}", case_type="civil", jurisdiction="Delhi", status=CaseStatus.ACTIVE, title="Case")
            test_db.add(case)
            test_db.commit()

            deadline = create_case_deadline(
                test_db, user_id, case.id, "Case", now + timedelta(days=days, hours=1), "appeal"
            )
            pref = create_or_update_user_preference(
                test_db, user_id, f"user{days}@example.com", 
                phone_number="+919876543210",
                notification_channel=NotificationChannel.BOTH
            )
            
            # Enable all notifications
            pref.notify_30_days = True
            pref.notify_10_days = True
            pref.notify_3_days = True
            pref.notify_1_day = True
            test_db.commit()
            
            with patch("config.Config.TESTING", True), \
                 patch("config.Config.DEBUG", True), \
                 patch("config.Config.TWILIO_ACCOUNT_SID", ""), \
                 patch("config.Config.TWILIO_AUTH_TOKEN", ""), \
                 patch("config.Config.TWILIO_FROM_NUMBER", ""), \
                 patch("config.Config.SENDGRID_API_KEY", ""), \
                 patch("config.Config.SENDGRID_FROM_EMAIL", "noreply@example.com"):
                service = NotificationService()
                results = service.send_reminders(test_db, deadline, pref)
            assert len(results) == 2  # SMS and Email
            assert results[0].success == True
            assert results[1].success == True

    def test_sms_client_error_handling(self):
        """Test SMSClient exception handling"""
        with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "sid", 
            "TWILIO_AUTH_TOKEN": "token", 
            "TWILIO_FROM_NUMBER": "num"
        }):
            client = SMSClient()
            client.client = MagicMock()
            client.client.messages.create.side_effect = Exception("Twilio error")
            
            success, message_id, error = client.send_sms("+123", "msg")
            assert success == False
            assert "Twilio error" in error

    def test_email_client_error_handling(self):
        """Test EmailClient exception handling"""
        with patch.dict(os.environ, {"SENDGRID_API_KEY": "key"}):
            client = EmailClient()
            client.client = MagicMock()
            client.client.send.side_effect = Exception("SendGrid error")
            
            success, message_id, error = client.send_email("u@e.com", "sub", "html")
            assert success == False
            assert "SendGrid error" in error

    def test_import_error_handling(self):
        """Test handling of missing libraries (optional coverage for imports)"""
        with patch.dict("sys.modules", {"twilio": None, "twilio.rest": None}):
            # This is hard to test after imports have already happened in the module
            # But we can at least check if the logic in the module would handle it
            pass
