"""
Tests for the deadline notification system.
Tests database models, notification services, and scheduler.
"""

import pytest
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import (
    Base,
    SessionLocal,
    NotificationStatus,
    NotificationChannel,
    CaseDeadline,
    UserPreference,
    NotificationLog,
    create_case_deadline,
    create_or_update_user_preference,
    get_upcoming_deadlines,
    has_notification_been_sent,
    log_notification,
    get_user_deadlines,
    get_notification_history,
)
from notification_service import (
    NotificationService,
    SMSClient,
    EmailClient,
    NotificationResult,
)
from scheduler import check_reminders_sync


# ==================== Database Setup for Testing ====================

@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


# ==================== Database Tests ====================

class TestDatabaseModels:
    """Test database models and ORM operations"""

    def test_create_case_deadline(self, test_db):
        """Test creating a case deadline"""
        deadline_date = datetime.now(timezone.utc) + timedelta(days=30)
        
        deadline = create_case_deadline(
            db=test_db,
            user_id="user123",
            case_id=1,
            case_title="Property Dispute",
            deadline_date=deadline_date,
            deadline_type="appeal",
            description="Appeal deadline",
        )

        assert deadline.user_id == "user123"
        assert deadline.case_id == 1
        assert deadline.case_title == "Property Dispute"
        assert deadline.is_completed == False
        assert deadline.days_until_deadline() >= 29  # Approximately 30 days

    def test_create_user_preference(self, test_db):
        """Test creating user notification preferences"""
        pref = create_or_update_user_preference(
            db=test_db,
            user_id="user123",
            email="user@example.com",
            phone_number="+91-9876543210",
            notification_channel=NotificationChannel.BOTH,
            timezone="Asia/Kolkata",
        )

        assert pref.user_id == "user123"
        assert pref.email == "user@example.com"
        assert pref.phone_number == "+91-9876543210"
        assert pref.notification_channel == NotificationChannel.BOTH
        assert pref.timezone == "Asia/Kolkata"

    def test_update_user_preference(self, test_db):
        """Test updating existing user preferences"""
        # Create initial preference
        create_or_update_user_preference(
            db=test_db,
            user_id="user123",
            email="old@example.com",
            phone_number="+91-1234567890",
        )

        # Update preference
        updated = create_or_update_user_preference(
            db=test_db,
            user_id="user123",
            email="new@example.com",
            phone_number="+91-9876543210",
            notification_channel=NotificationChannel.SMS,
        )

        assert updated.email == "new@example.com"
        assert updated.phone_number == "+91-9876543210"
        assert updated.notification_channel == NotificationChannel.SMS

    def test_get_upcoming_deadlines(self, test_db):
        """Test fetching upcoming deadlines"""
        now = datetime.now(timezone.utc)
        
        # Create deadlines at different time points
        create_case_deadline(
            test_db, "user1", 1, "Case 1",
            now + timedelta(days=5), "appeal"
        )
        create_case_deadline(
            test_db, "user1", 2, "Case 2",
            now + timedelta(days=15), "filing"
        )
        create_case_deadline(
            test_db, "user1", 3, "Case 3",
            now + timedelta(days=40), "submission"
        )

        # Get deadlines within 30 days
        upcoming = get_upcoming_deadlines(test_db, days_before=30)
        assert len(upcoming) == 2  # Should get cases 1 and 2

    def test_notification_logging(self, test_db):
        """Test logging notification attempts"""
        deadline = create_case_deadline(
            test_db, "user123", 1, "Case",
            datetime.now(timezone.utc) + timedelta(days=30), "appeal",
        )

        # Log SMS notification
        sms_log = log_notification(
            db=test_db,
            deadline_id=deadline.id,
            user_id="user123",
            channel=NotificationChannel.SMS,
            recipient="+91-9876543210",
            days_before=30,
            status=NotificationStatus.SENT,
            message_id="twilio_123",
        )

        assert sms_log.status == NotificationStatus.SENT
        assert sms_log.message_id == "twilio_123"
        assert sms_log.channel == NotificationChannel.SMS

    def test_prevent_duplicate_notifications(self, test_db):
        """Test that duplicate notifications are not sent"""
        deadline = create_case_deadline(
            test_db, "user123", 1, "Case",
            datetime.now(timezone.utc) + timedelta(days=30), "appeal",
        )

        # Log first notification
        log_notification(
            test_db, deadline.id, "user123", NotificationChannel.SMS,
            "+91-9876543210", 30, NotificationStatus.SENT,
        )

        # Check if already sent
        assert has_notification_been_sent(test_db, deadline.id, 30, NotificationChannel.SMS)

    def test_get_user_deadlines_sorted(self, test_db):
        """Test fetching user deadlines sorted by date"""
        now = datetime.now(timezone.utc)
        
        create_case_deadline(
            test_db, "user1", 1, "Case 1",
            now + timedelta(days=50), "appeal"
        )
        create_case_deadline(
            test_db, "user1", 2, "Case 2",
            now + timedelta(days=10), "filing"
        )

        deadlines = get_user_deadlines(test_db, "user1")
        assert len(deadlines) == 2
        assert deadlines[0].days_until_deadline() < deadlines[1].days_until_deadline()


# ==================== Notification Service Tests ====================

class TestNotificationService:
    """Test the notification service"""

    def test_sms_message_building(self):
        """Test SMS message format"""
        service = NotificationService()
        deadline_date = datetime.now(timezone.utc) + timedelta(days=10)
        
        message = service.build_sms_message("Property Dispute", 10, deadline_date)
        
        assert "LegalAssist" in message
        assert "Property Dispute" in message
        assert "10 day" in message
        assert len(message) <= 160  # Standard SMS length

    def test_email_message_building(self):
        """Test email message format"""
        service = NotificationService()
        deadline_date = datetime.now(timezone.utc) + timedelta(days=3)
        
        subject, html_content = service.build_email_message(
            "Appeal Filing", 3, deadline_date, "CASE-001"
        )
        
        assert "Urgent" in subject
        assert "Appeal Filing" in subject
        assert "3 day" in subject
        assert "CASE-001" in html_content
        assert "<html>" in html_content
        assert "deadline" in html_content.lower()

    @patch("notification_service.TwilioClient")
    def test_sms_send_success(self, mock_twilio, test_db):
        """Test successful SMS sending"""
        # Mock Twilio response
        mock_message = Mock()
        mock_message.sid = "SM123456789"
        mock_twilio.return_value.messages.create.return_value = mock_message

        # Create test data
        deadline = create_case_deadline(
            test_db, "user123", 1, "Test Case",
            datetime.now(timezone.utc) + timedelta(days=30), "appeal",
        )
        pref = create_or_update_user_preference(
            test_db, "user123", "user@example.com",
            phone_number="+91-9876543210",
        )

        # Mock environment variables
        with patch.dict(os.environ, {
            "TWILIO_ACCOUNT_SID": "test_sid",
            "TWILIO_AUTH_TOKEN": "test_token",
            "TWILIO_FROM_NUMBER": "+1234567890",
        }):
            service = NotificationService()
            result = service.send_sms_reminder(test_db, deadline, pref, 30)

        assert result.success == True
        assert result.channel == NotificationChannel.SMS
        assert result.message_id == "SM123456789"

    def test_sms_send_missing_phone(self, test_db):
        """Test SMS fails gracefully when no phone number"""
        deadline = create_case_deadline(
            test_db, "user123", 1, "Test Case",
            datetime.now(timezone.utc) + timedelta(days=30), "appeal",
        )
        pref = create_or_update_user_preference(
            test_db, "user123", "user@example.com",
            phone_number=None,  # No phone
        )

        service = NotificationService()
        result = service.send_sms_reminder(test_db, deadline, pref, 30)

        assert result.success == False
        assert "phone number" in result.error.lower()

    @patch("notification_service.SendGridAPIClient")
    def test_email_send_success(self, mock_sendgrid, test_db):
        """Test successful email sending"""
        # Mock SendGrid response
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {"X-Message-ID": "email_123"}
        mock_sendgrid.return_value.send.return_value = mock_response

        deadline = create_case_deadline(
            test_db, "user123", 1, "Test Case",
            datetime.now(timezone.utc) + timedelta(days=10), "appeal",
        )
        pref = create_or_update_user_preference(
            test_db, "user123", "user@example.com",
        )

        with patch.dict(os.environ, {
            "SENDGRID_API_KEY": "test_key",
            "SENDGRID_FROM_EMAIL": "noreply@legalassist.ai",
        }):
            service = NotificationService()
            result = service.send_email_reminder(test_db, deadline, pref, 10)

        assert result.success == True
        assert result.channel == NotificationChannel.EMAIL

    def test_mock_mode_sms(self, test_db):
        """Test SMS in mock mode (no credentials)"""
        deadline = create_case_deadline(
            test_db, "user123", 1, "Test Case",
            datetime.now(timezone.utc) + timedelta(days=30), "appeal",
        )
        pref = create_or_update_user_preference(
            test_db, "user123", "user@example.com",
            phone_number="+91-9876543210",
        )

        # Clear environment variables to trigger mock mode
        with patch.dict(os.environ, {}, clear=True):
            service = NotificationService()
            result = service.send_sms_reminder(test_db, deadline, pref, 30)

        # Mock mode should still return success
        assert result.success == True
        assert "mock_sms" in result.message_id

    def test_mock_mode_email(self, test_db):
        """Test email in mock mode (no API key)"""
        deadline = create_case_deadline(
            test_db, "user123", 1, "Test Case",
            datetime.now(timezone.utc) + timedelta(days=10), "appeal",
        )
        pref = create_or_update_user_preference(
            test_db, "user123", "user@example.com",
        )

        with patch.dict(os.environ, {}, clear=True):
            service = NotificationService()
            result = service.send_email_reminder(test_db, deadline, pref, 10)

        assert result.success == True
        assert "mock_email" in result.message_id


# ==================== Scheduler Tests ====================

class TestScheduler:
    """Test the background scheduler"""

    def test_sync_reminder_check_basic(self, test_db):
        """Test synchronous reminder check"""
        now = datetime.now(timezone.utc)
        
        # Create deadline at exactly 30 days
        create_case_deadline(
            test_db, "user1", 1, "Case 1",
            now + timedelta(days=30), "appeal",
        )
        
        # Create user preference
        create_or_update_user_preference(
            test_db, "user1", "user@example.com",
            phone_number="+91-9876543210",
        )

        # Mock the notification service to count calls
        with patch("scheduler.notification_service") as mock_service, \
             patch("scheduler.SessionLocal", return_value=test_db):
            mock_service.send_reminders.return_value = []
            check_reminders_sync(target_days=30)

    def test_sync_reminder_respects_preferences(self, test_db):
        """Test that reminders respect user preferences"""
        now = datetime.now(timezone.utc)
        
        deadline = create_case_deadline(
            test_db, "user1", 1, "Case 1",
            now + timedelta(days=30), "appeal",
        )
        
        # Create preference with 30-day reminder disabled
        pref = create_or_update_user_preference(
            test_db, "user1", "user@example.com",
            phone_number="+91-9876543210",
        )
        pref.notify_30_days = False
        test_db.commit()

        # Verify preference was saved
        check_pref = test_db.query(UserPreference).filter_by(user_id="user1").first()
        assert check_pref.notify_30_days == False


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for the full notification flow"""

    def test_complete_notification_flow(self, test_db):
        """Test complete flow: deadline -> preference -> notification"""
        # 1. Create deadline
        deadline_date = datetime.now(timezone.utc) + timedelta(days=30)
        deadline = create_case_deadline(
            test_db, "user1", 1, "Appeal Filing",
            deadline_date, "appeal", "Need to submit appeal"
        )

        # 2. Create user preference
        pref = create_or_update_user_preference(
            test_db, "user1", "user@example.com",
            phone_number="+91-9876543210",
            notification_channel=NotificationChannel.BOTH,
            timezone="Asia/Kolkata",
        )

        # 3. Mock notification sending
        with patch.dict(os.environ, {}, clear=True):
            service = NotificationService()
            
            # Send SMS
            sms_result = service.send_sms_reminder(test_db, deadline, pref, 30)
            assert sms_result.success == True
            
            # Send Email
            email_result = service.send_email_reminder(test_db, deadline, pref, 30)
            assert email_result.success == True

        # 4. Verify logs were created
        logs = get_notification_history(test_db, "user1")
        assert len(logs) >= 2
        
        # Verify we can't send duplicates
        assert has_notification_been_sent(test_db, deadline.id, 30, NotificationChannel.SMS)
        assert has_notification_been_sent(test_db, deadline.id, 30, NotificationChannel.EMAIL)

    def test_timezone_awareness(self, test_db):
        """Test that timezone preferences are stored and retrieved"""
        timezones_to_test = [
            "UTC",
            "Asia/Kolkata",
            "America/New_York",
            "Europe/London",
        ]

        for tz in timezones_to_test:
            pref = create_or_update_user_preference(
                test_db, f"user_{tz}", f"user_{tz}@example.com",
                timezone=tz,
            )
            assert pref.timezone == tz

    def test_multiple_reminders_same_deadline(self, test_db):
        """Test that all reminder thresholds work for same deadline"""
        now = datetime.now(timezone.utc)
        deadline = create_case_deadline(
            test_db, "user1", 1, "Case",
            now + timedelta(days=30), "appeal",
        )
        pref = create_or_update_user_preference(
            test_db, "user1", "user@example.com",
            phone_number="+91-9876543210",
        )

        # Log reminders at different thresholds
        for days in [30, 10, 3, 1]:
            log_notification(
                test_db, deadline.id, "user1", NotificationChannel.SMS,
                "+91-9876543210", days, NotificationStatus.SENT,
            )

        # Verify all were logged
        for days in [30, 10, 3, 1]:
            assert has_notification_been_sent(
                test_db, deadline.id, days, NotificationChannel.SMS
            )


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
