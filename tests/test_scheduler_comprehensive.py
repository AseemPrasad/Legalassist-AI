
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
    Case,
    CaseStatus,
    User,
)
from notification_service import NotificationResult
from scheduler import (
    check_and_send_reminders,
    _scheduler,
    start_scheduler,
    stop_scheduler,
    trigger_reminder_check_now,
    check_reminders_sync,
    get_scheduler,
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
            user_id = days
            user = User(id=user_id, email=f"user{days}@example.com")
            test_db.add(user)
            test_db.commit()

            case_id_int = 100 + days
            case = Case(user_id=user_id, case_number=f"CASE-{case_id_int}", case_type="civil", jurisdiction="Delhi", status=CaseStatus.ACTIVE, title=f"Title {days}")
            test_db.add(case)
            test_db.commit()

            deadline_date = now + timedelta(days=days, hours=1)
            create_case_deadline(
                test_db, user_id, case.id, f"Title {days}",
                deadline_date, "appeal"
            )
            create_or_update_user_preference(
                test_db, user_id, f"user{days}@example.com",
                phone_number=f"+91{days}00000000",
                notification_channel=NotificationChannel.BOTH
            )

        # Mock dependencies and underlying send functions (not the whole service)
        with patch("scheduler.SessionLocal", return_value=test_db), \
             patch("scheduler.notification_service.send_reminders") as mock_send_reminders, \
             patch("scheduler.is_reminder_time_for_user", return_value=True):
            
            # send_reminders returns a list of NotificationResult objects
            mock_send_reminders.return_value = [
                NotificationResult(success=True, channel=NotificationChannel.SMS, recipient="+91test", message_id="sms_123", error=None),
                NotificationResult(success=True, channel=NotificationChannel.EMAIL, recipient="test@example.com", message_id="email_123", error=None),
            ]
            
            check_and_send_reminders()
            
            # Verify it called send_reminders 4 times (once per threshold)
            assert mock_send_reminders.call_count == 4

    def test_check_and_send_reminders_no_preferences(self, test_db):
        """Test when user has no preferences"""
        now = datetime.now(timezone.utc)
        
        user = User(id=1, email="nopref@example.com")
        test_db.add(user)
        test_db.commit()

        case = Case(user_id=1, case_number="CASE-999", case_type="civil", jurisdiction="Delhi", status=CaseStatus.ACTIVE)
        test_db.add(case)
        test_db.commit()

        create_case_deadline(
            test_db, 1, case.id, "Title",
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
        # Mock setup_scheduler to avoid creating real schedulers
        with patch("scheduler.setup_scheduler") as mock_setup, \
             patch("scheduler._scheduler", None):
            
            mock_sched = MagicMock()
            mock_sched.running = False
            mock_setup.return_value = mock_sched
            
            # Test start_scheduler
            start_scheduler()
            assert mock_sched.start.called
            
            # Test stop_scheduler
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
        deadline_date = now + timedelta(days=30, hours=1)
        user = User(id=1, email="user@example.com")
        test_db.add(user)
        test_db.commit()

        case = Case(user_id=1, case_number="CASE-1", case_type="civil", jurisdiction="Delhi", status=CaseStatus.ACTIVE)
        test_db.add(case)
        test_db.commit()

        create_case_deadline(
            test_db, 1, case.id, "Title",
            deadline_date, "appeal"
        )
        create_or_update_user_preference(
            test_db, 1, "user@example.com",
            phone_number="+911234567890"
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
