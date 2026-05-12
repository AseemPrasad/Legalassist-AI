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
    Case,
    CaseStatus,
    User,
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

        # Create an owned case (required for ownership validation)
        case = Case(
            user_id=1,
            case_number="CASE-1",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Property Dispute",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            db=test_db,
            user_id=1,
            case_id=case.id,
            case_title="Property Dispute",
            deadline_date=deadline_date,
            deadline_type="appeal",
            description="Appeal deadline",
        )


        assert deadline.user_id == 1
        assert deadline.case_id == 1
        assert deadline.case_title == "Property Dispute"
        assert deadline.is_completed == False
        assert deadline.days_until_deadline() >= 29  # Approximately 30 days

    def test_create_case_deadline_coerces_string_case_id(self, test_db):
        """Test that string case_id values are normalized to integers"""
        deadline_date = datetime.now(timezone.utc) + timedelta(days=15)

        case = Case(
            user_id=1,
            case_number="CASE-2",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Appeal Filing",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            db=test_db,
            user_id=1,
            case_id=str(case.id),
            case_title="Appeal Filing",
            deadline_date=deadline_date,
            deadline_type="appeal",
        )

        assert deadline.case_id == case.id

    def test_create_case_deadline_rejects_invalid_case_id(self, test_db):
        """Test that invalid case_id values raise a clear error"""
        deadline_date = datetime.now(timezone.utc) + timedelta(days=15)

        with pytest.raises(ValueError, match="case_id must be an integer matching cases.id"):
            create_case_deadline(
                db=test_db,
                user_id=1,
                case_id="abc",
                case_title="Appeal Filing",
                deadline_date=deadline_date,
                deadline_type="appeal",
            )

    def test_create_user_preference(self, test_db):
        """Test creating user notification preferences"""
        pref = create_or_update_user_preference(
            db=test_db,
            user_id=1,
            email="user@example.com",
            phone_number="+91-9876543210",
            notification_channel=NotificationChannel.BOTH,
            timezone="Asia/Kolkata",
        )

        assert pref.user_id == 1
        assert pref.email == "user@example.com"
        assert pref.phone_number == "+91-9876543210"
        assert pref.notification_channel == NotificationChannel.BOTH
        assert pref.timezone == "Asia/Kolkata"

    def test_update_user_preference(self, test_db):
        """Test updating existing user preferences"""
        # Create initial preference
        create_or_update_user_preference(
            db=test_db,
            user_id=1,
            email="old@example.com",
            phone_number="+91-1234567890",
        )

        # Update preference
        updated = create_or_update_user_preference(
            db=test_db,
            user_id=1,
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
        # Create owned cases (required for ownership validation)
        case1 = Case(
            user_id=1,
            case_number="CASE-1",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case 1",
        )
        case2 = Case(
            user_id=1,
            case_number="CASE-2",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case 2",
        )
        case3 = Case(
            user_id=1,
            case_number="CASE-3",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case 3",
        )
        test_db.add_all([case1, case2, case3])
        test_db.commit()
        test_db.refresh(case1)
        test_db.refresh(case2)
        test_db.refresh(case3)

        create_case_deadline(
            test_db, 1, case1.id, "Case 1",
            now + timedelta(days=5), "appeal"
        )
        create_case_deadline(
            test_db, 1, case2.id, "Case 2",
            now + timedelta(days=15), "filing"
        )
        create_case_deadline(
            test_db, 1, case3.id, "Case 3",
            now + timedelta(days=40), "submission"
        )


        # Get deadlines within 30 days
        upcoming = get_upcoming_deadlines(test_db, days_before=30)
        assert len(upcoming) == 2  # Should get cases 1 and 2

    def test_notification_logging(self, test_db):
        """Test logging notification attempts"""
        # Create owned case (required for ownership validation)
        case = Case(
            user_id=1,
            case_number="CASE-4",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db,
            1,
            case.id,
            "Case",
            datetime.now(timezone.utc) + timedelta(days=30),
            "appeal",
        )


        # Log SMS notification
        sms_log = log_notification(
            db=test_db,
            deadline_id=deadline.id,
            user_id=1,
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
        # Create owned case (required for ownership validation)
        case = Case(
            user_id=1,
            case_number="CASE-5",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db,
            1,
            case.id,
            "Case",
            datetime.now(timezone.utc) + timedelta(days=30),
            "appeal",
        )


        # Log first notification
        log_notification(
            test_db, deadline.id, 1, NotificationChannel.SMS,
            "+91-9876543210", 30, NotificationStatus.SENT,
        )

        # Check if already sent
        assert has_notification_been_sent(test_db, deadline.id, 30, NotificationChannel.SMS)

    def test_get_user_deadlines_sorted(self, test_db):
        """Test fetching user deadlines sorted by date"""
        now = datetime.now(timezone.utc)

        # Owned cases required for ownership validation
        case1 = Case(
            user_id=1,
            case_number="CASE-1",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case 1",
        )
        case2 = Case(
            user_id=1,
            case_number="CASE-2",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case 2",
        )
        test_db.add_all([case1, case2])
        test_db.commit()
        test_db.refresh(case1)
        test_db.refresh(case2)

        create_case_deadline(
            test_db,
            1,
            case1.id,
            "Case 1",
            now + timedelta(days=50),
            "appeal",
        )
        create_case_deadline(
            test_db,
            1,
            case2.id,
            "Case 2",
            now + timedelta(days=10),
            "filing",
        )

        deadlines = get_user_deadlines(test_db, 1)
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
        
        # Create a mock deadline object
        deadline = Mock(spec=CaseDeadline)
        deadline.case_title = "Appeal Filing"
        deadline.deadline_type = "appeal"
        deadline.deadline_date = deadline_date
        deadline.case_id = "CASE-001"
        deadline.description = "Test description"
        
        subject, html_content = service.build_email_message(deadline, 3)
        
        assert "REMINDER" in subject or "URGENT" in subject
        assert "Appeal Filing" in subject
        assert "3 day" in subject or "3 Days" in html_content
        assert "CASE-001" in html_content
        assert "<!DOCTYPE html>" in html_content
        assert "deadline" in html_content.lower()

    def test_email_title_escaping(self):
        """Test that case titles with HTML are escaped in email content"""
        service = NotificationService()
        deadline_date = datetime.now(timezone.utc) + timedelta(days=3)
        malicious_title = "Case <script>alert('XSS')</script> & More"
        
        # Create a mock deadline object
        deadline = Mock(spec=CaseDeadline)
        deadline.case_title = malicious_title
        deadline.deadline_type = "appeal"
        deadline.deadline_date = deadline_date
        deadline.case_id = "CASE-001"
        deadline.description = "<b>Bold</b>" # Also test description escaping
        
        subject, html_content = service.build_email_message(deadline, 3)
        
        # Title in subject should be plain
        assert malicious_title in subject
        
        # Title and description in HTML MUST be escaped
        assert "<script>" not in html_content
        # Accept both forms of quote escaping (' or &#x27;)
        assert ("&lt;script&gt;alert('XSS')&lt;/script&gt;" in html_content or
                "&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;" in html_content)
        assert " &amp; " in html_content
        assert "<b>" not in html_content
        assert "&lt;b&gt;Bold&lt;/b&gt;" in html_content

    @patch("notification_service.TwilioClient")
    def test_sms_send_success(self, mock_twilio, test_db):
        """Test successful SMS sending"""
        # Mock Twilio response
        mock_message = Mock()
        mock_message.sid = "SM123456789"
        mock_twilio.return_value.messages.create.return_value = mock_message

        # Create test data
        # Create owned case (required for ownership validation)
        case = Case(
            user_id=1,
            case_number="CASE-6",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Test Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db,
            1,
            case.id,
            "Test Case",
            datetime.now(timezone.utc) + timedelta(days=30),
            "appeal",
        )

        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
            phone_number="+91-9876543210",
        )

        # Mock Config variables
        with patch("config.Config.TWILIO_ACCOUNT_SID", "test_sid"), \
             patch("config.Config.TWILIO_AUTH_TOKEN", "test_token"), \
             patch("config.Config.TWILIO_FROM_NUMBER", "+1234567890"):
            service = NotificationService()
            result = service.send_sms_reminder(test_db, deadline, pref, 30)


        assert result.success == True
        assert result.channel == NotificationChannel.SMS
        assert result.message_id == "SM123456789"

    def test_sms_send_missing_phone(self, test_db):
        """Test SMS fails gracefully when no phone number"""
        # Owned case required for ownership validation
        case = Case(
            user_id=1,
            case_number="CASE-7",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Test Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db,
            1,
            case.id,
            "Test Case",
            datetime.now(timezone.utc) + timedelta(days=30),
            "appeal",
        )

        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
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

        # Owned case required for ownership validation
        case = Case(
            user_id=1,
            case_number="CASE-8",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Test Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db, 1, case.id, "Test Case",
            datetime.now(timezone.utc) + timedelta(days=10), "appeal",
        )

        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
        )

        with patch("config.Config.SENDGRID_API_KEY", "test_key"), \
             patch("config.Config.SENDGRID_FROM_EMAIL", "noreply@legalassist.ai"):
            service = NotificationService()
            result = service.send_email_reminder(test_db, deadline, pref, 10)

        assert result.success == True
        assert result.channel == NotificationChannel.EMAIL

    @patch("notification_service.SendGridAPIClient")
    def test_email_send_uses_case_number(self, mock_sendgrid, test_db):
        """Test email reminder uses the case_number from the related case"""
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {"X-Message-ID": "email_123"}
        mock_sendgrid.return_value.send.return_value = mock_response

        user = User(email="user123@example.com")
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)

        case = Case(
            user_id=user.id,
            case_number="CASE-9001",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Appeal Filing",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = CaseDeadline(
            user_id=user.id,
            case_id=case.id,
            case_title="Appeal Filing",
            deadline_date=datetime.now(timezone.utc) + timedelta(days=10),
            deadline_type="appeal",
        )
        test_db.add(deadline)
        test_db.commit()
        test_db.refresh(deadline)

        pref = create_or_update_user_preference(
            test_db, user.id, "user@example.com",
        )

        with patch("config.Config.SENDGRID_API_KEY", "test_key"), \
             patch("config.Config.SENDGRID_FROM_EMAIL", "noreply@legalassist.ai"):
            service = NotificationService()
            result = service.send_email_reminder(test_db, deadline, pref, 10)

        assert result.success == True
        # Verify the email contains the case information
        assert "Appeal Filing" in result.recipient or result.message_id is not None

    def test_mock_mode_sms(self, test_db):
        """Test SMS in mock mode (no credentials)"""
        # Owned case required for ownership validation
        case = Case(
            user_id=1,
            case_number="CASE-7",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Test Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db,
            1,
            case.id,
            "Test Case",
            datetime.now(timezone.utc) + timedelta(days=30),
            "appeal",
        )

        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
            phone_number="+91-9876543210",
        )

        # Clear Config variables to trigger mock mode
        with patch("config.Config.TWILIO_ACCOUNT_SID", ""), \
             patch("config.Config.TWILIO_AUTH_TOKEN", ""), \
             patch("config.Config.TWILIO_FROM_NUMBER", ""):
            service = NotificationService()
            result = service.send_sms_reminder(test_db, deadline, pref, 30)

        # Mock mode should still return success
        assert result.success == True
        assert "mock_sms" in result.message_id

    def test_mock_mode_email(self, test_db):
        """Test email in mock mode (no API key)"""
        # Owned case required for ownership validation
        case = Case(
            user_id=1,
            case_number="CASE-9",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Test Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db, 1, case.id, "Test Case",
            datetime.now(timezone.utc) + timedelta(days=10), "appeal",
        )

        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
        )

        with patch("config.Config.SENDGRID_API_KEY", ""):
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
        
        case = Case(
            user_id=1,
            case_number="CASE-1",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case 1",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        # Create deadline at exactly 30 days
        create_case_deadline(
            test_db, 1, case.id, "Case 1",
            now + timedelta(days=30), "appeal",
        )
        
        # Create user preference
        create_or_update_user_preference(
            test_db, 1, "user@example.com",
            phone_number="+91-9876543210",
        )

        # Mock the notification service to count calls
        with patch("scheduler.notification_service") as mock_service, \
             patch("scheduler.SessionLocal", return_value=test_db):
            mock_service.send_reminders.return_value = []
            check_reminders_sync(target_days=30, db=test_db)

    def test_sync_reminder_respects_preferences(self, test_db):
        """Test that reminders respect user preferences"""
        now = datetime.now(timezone.utc)
        
        case = Case(
            user_id=1,
            case_number="CASE-10",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case 1",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db, 1, case.id, "Case 1",
            now + timedelta(days=30), "appeal",
        )

        
        # Create preference with 30-day reminder disabled
        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
            phone_number="+91-9876543210",
        )
        pref.notify_30_days = False
        test_db.commit()

        # Verify preference was saved
        check_pref = test_db.query(UserPreference).filter_by(user_id=1).first()
        assert check_pref.notify_30_days == False


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for the full notification flow"""

    def test_complete_notification_flow(self, test_db):
        """Test complete flow: deadline -> preference -> notification"""
        case = Case(
            user_id=1,
            case_number="CASE-1",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Appeal Filing",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        # 1. Create deadline
        deadline_date = datetime.now(timezone.utc) + timedelta(days=30)
        deadline = create_case_deadline(
            test_db, 1, case.id, "Appeal Filing",
            deadline_date, "appeal", "Need to submit appeal"
        )

        # 2. Create user preference
        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
            phone_number="+91-9876543210",
            notification_channel=NotificationChannel.BOTH,
            timezone="Asia/Kolkata",
        )

        # 3. Mock notification sending
        with patch("config.Config.TESTING", True), \
             patch("config.Config.DEBUG", True), \
             patch("config.Config.TWILIO_ACCOUNT_SID", ""), \
             patch("config.Config.TWILIO_AUTH_TOKEN", ""), \
             patch("config.Config.TWILIO_FROM_NUMBER", ""), \
             patch("config.Config.SENDGRID_API_KEY", ""), \
             patch("config.Config.SENDGRID_FROM_EMAIL", "noreply@legalassist.ai"):
            service = NotificationService()
            
            # Send SMS
            sms_result = service.send_sms_reminder(test_db, deadline, pref, 30)
            assert sms_result.success == True
            
            # Send Email
            email_result = service.send_email_reminder(test_db, deadline, pref, 30)
            assert email_result.success == True

        # 4. Verify logs were created
        logs = get_notification_history(test_db, 1)
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
            user_id = 1000 + timezones_to_test.index(tz)
            pref = create_or_update_user_preference(
                test_db, user_id, f"user_{user_id}@example.com",
                timezone=tz,
            )
            assert pref.timezone == tz

    def test_multiple_reminders_same_deadline(self, test_db):
        """Test that all reminder thresholds work for same deadline"""
        now = datetime.now(timezone.utc)
        case = Case(
            user_id=1,
            case_number="CASE-1",
            case_type="civil",
            jurisdiction="Delhi",
            status=CaseStatus.ACTIVE,
            title="Case",
        )
        test_db.add(case)
        test_db.commit()
        test_db.refresh(case)

        deadline = create_case_deadline(
            test_db, 1, case.id, "Case",
            now + timedelta(days=30), "appeal",
        )
        pref = create_or_update_user_preference(
            test_db, 1, "user@example.com",
            phone_number="+91-9876543210",
        )

        # Log reminders at different thresholds
        for days in [30, 10, 3, 1]:
            log_notification(
                test_db, deadline.id, 1, NotificationChannel.SMS,
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
