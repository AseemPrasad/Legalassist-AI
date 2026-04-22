
import pytest
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
)
from scheduler import (
    check_and_send_reminders,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
    trigger_reminder_check_now,
    check_reminders_sync,
)

@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory test database"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()

class TestSchedulerComprehensive:
    """Comprehensive tests for the scheduler module"""

    def test_check_and_send_reminders_flow(self, test_db):
        """Test the main check_and_send_reminders function"""
        now = datetime.now(timezone.utc)
        
        # Create deadlines at threshold days
        for days in [30, 10, 3, 1]:
            create_case_deadline(
                test_db, f"user_{days}", f"CASE_{days}", f"Title {days}",
                now + timedelta(days=days, minutes=5), "appeal"
            )
            create_or_update_user_preference(
                test_db, f"user_{days}", f"user{days}@example.com",
                phone_number=f"+91{days}00000000",
                notification_channel=NotificationChannel.BOTH
            )

        # Mock dependencies
        with patch("scheduler.SessionLocal", return_value=test_db), \
             patch("scheduler.notification_service") as mock_service:
            
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.recipient = "test"
            mock_service.send_sms_reminder.return_value = mock_result
            mock_service.send_email_reminder.return_value = mock_result
            
            check_and_send_reminders()
            
            # Verify it processed all 4 thresholds
            assert mock_service.send_sms_reminder.call_count == 4
            assert mock_service.send_email_reminder.call_count == 4

    def test_check_and_send_reminders_no_preferences(self, test_db):
        """Test when user has no preferences"""
        now = datetime.now(timezone.utc)
        create_case_deadline(
            test_db, "no_pref_user", "CASE_001", "Title",
            now + timedelta(days=30, minutes=5), "appeal"
        )
        # No preference created

        with patch("scheduler.SessionLocal", return_value=test_db), \
             patch("scheduler.notification_service") as mock_service:
            check_and_send_reminders()
            assert mock_service.send_sms_reminder.call_count == 0

    def test_get_scheduler_initialization(self):
        """Test scheduler singleton initialization"""
        with patch("scheduler.BackgroundScheduler") as mock_sched_class:
            mock_sched = mock_sched_class.return_value
            # Reset global state
            with patch("scheduler._scheduler", None):
                s = get_scheduler()
                assert s == mock_sched
                assert mock_sched.add_job.called

    def test_start_stop_scheduler(self):
        """Test starting and stopping the scheduler"""
        with patch("scheduler.get_scheduler") as mock_get:
            mock_sched = mock_get.return_value
            mock_sched.running = False
            
            start_scheduler()
            assert mock_sched.start.called
            
            mock_sched.running = True
            with patch("scheduler._scheduler", mock_sched):
                stop_scheduler()
                assert mock_sched.shutdown.called

    def test_trigger_now(self):
        """Test manual trigger"""
        with patch("scheduler.check_and_send_reminders") as mock_check:
            trigger_reminder_check_now()
            assert mock_check.called

    def test_check_reminders_sync_target_days(self, test_db):
        """Test sync version with target days filter"""
        now = datetime.now(timezone.utc)
        create_case_deadline(
            test_db, "user1", "CASE_1", "Title",
            now + timedelta(days=30, minutes=5), "appeal"
        )
        create_or_update_user_preference(
            test_db, "user1", "user@example.com"
        )
        
        with patch("scheduler.SessionLocal", return_value=test_db), \
             patch("scheduler.notification_service") as mock_service:
            mock_service.send_reminders.return_value = [MagicMock(success=True)]
            
            # Target 1 day (should not find the 30 day deadline)
            count = check_reminders_sync(target_days=1)
            assert count == 0
            
            # Target 30 days
            count = check_reminders_sync(target_days=30)
            assert count == 1
