
import pytest
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from sqlalchemy import event
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from database import (
    Base,
    CaseDeadline,
    UserPreference,
    NotificationChannel,
    create_case_deadline,
    create_or_update_user_preference,
    log_notification,
    Case,
    CaseStatus,
    User,
    NotificationStatus,
)
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
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email VARCHAR(255) NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE cases (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                case_number VARCHAR(255) NOT NULL,
                case_type VARCHAR(255) NOT NULL,
                jurisdiction VARCHAR(255) NOT NULL,
                status VARCHAR(50) NOT NULL,
                title VARCHAR(255),
                version INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME,
                updated_at DATETIME
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE case_deadlines (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                case_id INTEGER NOT NULL,
                case_title VARCHAR(255) NOT NULL,
                deadline_date DATETIME NOT NULL,
                deadline_type VARCHAR(255) NOT NULL,
                description TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                is_completed BOOLEAN DEFAULT 0 NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE user_preferences (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL UNIQUE,
                phone_number VARCHAR(255),
                email VARCHAR(255) NOT NULL,
                notification_channel VARCHAR(50) DEFAULT 'both',
                timezone VARCHAR(255) DEFAULT 'UTC',
                notify_30_days BOOLEAN DEFAULT 1,
                notify_10_days BOOLEAN DEFAULT 1,
                notify_3_days BOOLEAN DEFAULT 1,
                notify_1_day BOOLEAN DEFAULT 1,
                holiday_aware_reminders BOOLEAN DEFAULT 0,
                holiday_country VARCHAR(255),
                holiday_region VARCHAR(255),
                holiday_calendar_json TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE notification_logs (
                id INTEGER PRIMARY KEY,
                deadline_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                channel VARCHAR(50) NOT NULL,
                status VARCHAR(50),
                recipient VARCHAR(255) NOT NULL,
                days_before INTEGER NOT NULL,
                message_id VARCHAR(255),
                error_message TEXT,
                message_preview TEXT,
                sent_at DATETIME,
                delivered_at DATETIME,
                failed_at DATETIME,
                created_at DATETIME
            )
            """
        )
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
            with patch("scheduler.init_db"), \
                 patch("scheduler.SessionLocal", return_value=test_db), \
                 patch("scheduler.notification_service.send_reminders") as mock_send_reminders, \
                 patch("scheduler.is_reminder_time_for_user", return_value=True):

        # Mock dependencies and underlying send functions (not the whole service)
        with patch("scheduler.SessionLocal", return_value=test_db), \
             patch("scheduler.notification_service.send_reminders") as mock_send_reminders, \
             patch("scheduler.is_reminder_time_for_user", return_value=True):
            
            # send_reminders returns a list of NotificationResult objects
            mock_send_reminders.return_value = [
                SimpleNamespace(success=True, channel=NotificationChannel.SMS, recipient="+91test", message_id="sms_123", error=None),
                SimpleNamespace(success=True, channel=NotificationChannel.EMAIL, recipient="test@example.com", message_id="email_123", error=None),
            ]
            
            check_and_send_reminders()
            
            # Verify it called send_reminders 4 times (once per threshold)
            assert mock_send_reminders.call_count == 4

    def test_check_and_send_reminders_continues_after_send_failure(self):
        """Test that one deadline failure does not stop later deadlines from being processed."""
        now = datetime.now(timezone.utc)

        deadlines = [
            SimpleNamespace(id=1, user_id=1, case_id=101, days_until_deadline=lambda: 30),
            SimpleNamespace(id=2, user_id=2, case_id=102, days_until_deadline=lambda: 30),
        ]
        prefs = [
            SimpleNamespace(user_id=1, timezone="UTC", notification_channel=NotificationChannel.SMS, email="user1@example.com", phone_number="+911000000001", notify_30_days=True, notify_10_days=False, notify_3_days=False, notify_1_day=False),
            SimpleNamespace(user_id=2, timezone="UTC", notification_channel=NotificationChannel.SMS, email="user2@example.com", phone_number="+911000000002", notify_30_days=True, notify_10_days=False, notify_3_days=False, notify_1_day=False),
        ]

        class FakeQuery:
            def __init__(self, results):
                self._results = results

            def filter(self, *args, **kwargs):
                return self

            def all(self):
                return self._results

        class FakeDb:
            def query(self, model):
                return FakeQuery(prefs)

            def close(self):
                return None

        fake_db = FakeDb()

        call_count = {"value": 0}

        def fake_send_reminders(db, deadline, user_preference, days_left):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise RuntimeError("boom")
            return [
                SimpleNamespace(
                    success=True,
                    channel=NotificationChannel.SMS,
                    recipient=user_preference.phone_number,
                    message_id="sms_123",
                    error=None,
                )
            ]

        with patch("scheduler.init_db"), \
             patch("scheduler.SessionLocal", return_value=fake_db), \
             patch("scheduler.get_reminder_dispatch_candidates", return_value=[
                 (deadlines[0], 30, prefs[0]),
                 (deadlines[1], 30, prefs[1]),
             ]), \
             patch("scheduler.notification_service.send_reminders", side_effect=fake_send_reminders) as mock_send_reminders, \
             patch("scheduler.logger.error") as mock_error:
            check_and_send_reminders()

        assert mock_send_reminders.call_count == 2
        assert mock_error.call_count == 1
        assert mock_error.call_args.kwargs["deadline_id"] is not None
        assert mock_error.call_args.kwargs["user_id"] is not None
        assert mock_error.call_args.kwargs["days_left"] == 30
        assert "boom" in mock_error.call_args.kwargs["error"]

    def test_check_and_send_reminders_survives_helper_exception(self):
        """Test that the scheduler job returns cleanly when the send helper raises."""

        deadline = SimpleNamespace(
            id=1,
            user_id=1,
            case_id=101,
            days_until_deadline=lambda: 30,
        )
        preference = SimpleNamespace(
            user_id=1,
            timezone="UTC",
            notification_channel=NotificationChannel.BOTH,
            email="user@example.com",
            phone_number="+911234567890",
            notify_30_days=True,
            notify_10_days=True,
            notify_3_days=True,
            notify_1_day=True,
        )

        class FakeQuery:
            def filter(self, *args, **kwargs):
                return self

            def all(self):
                return [preference]

        class FakeDb:
            def query(self, model):
                return FakeQuery()

            def close(self):
                return None

        @contextmanager
        def locked():
            yield True

        with patch("scheduler.distributed_lock", return_value=locked()), \
             patch("scheduler.init_db"), \
             patch("scheduler.SessionLocal", return_value=FakeDb()), \
             patch("scheduler.get_upcoming_deadlines", return_value=[deadline]), \
             patch("scheduler.should_process_threshold", return_value=True), \
             patch("scheduler.is_notify_enabled", return_value=True), \
             patch("scheduler.is_reminder_time_for_user", return_value=True), \
             patch("scheduler._send_deadline_reminders_safe", side_effect=RuntimeError("boom")):
            count = check_and_send_reminders()

        assert count == 0

    def test_run_system_maintenance_task_disabled_by_default(self):
        """Test that maintenance tasks are skipped unless explicitly enabled."""
        import scheduler

        with patch.object(scheduler, "ENABLE_MAINTENANCE_TASKS", False), \
             patch.object(scheduler, "logger") as mock_logger, \
             patch.object(scheduler, "distributed_lock") as mock_lock:
            mock_lock.return_value.__enter__.return_value = True
            mock_lock.return_value.__exit__.return_value = False

            scheduler.run_system_maintenance_task()

        mock_logger.info.assert_any_call("scheduler_maintenance_disabled")

    def test_run_system_maintenance_task_uses_configured_command(self):
        """Test that enabled maintenance runs the configured command instead of a hard-coded example."""
        import scheduler

        fake_process = MagicMock()
        fake_process.communicate.return_value = ("ok", "")
        fake_process.returncode = 0

        captured_command = {}

        @contextmanager
        def fake_managed_subprocess(command, **kwargs):
            captured_command["value"] = command
            yield fake_process

        with patch.object(scheduler, "ENABLE_MAINTENANCE_TASKS", True), \
               patch.object(scheduler, "MAINTENANCE_TASK_COMMAND", "python -m maintenance_runner"), \
             patch.object(scheduler, "managed_subprocess", fake_managed_subprocess), \
             patch.object(scheduler, "logger") as mock_logger, \
             patch.object(scheduler, "distributed_lock") as mock_lock:
            mock_lock.return_value.__enter__.return_value = True
            mock_lock.return_value.__exit__.return_value = False

            scheduler.run_system_maintenance_task()

        assert captured_command["value"] == ["python", "-m", "maintenance_runner"]
        mock_process_communicate = fake_process.communicate
        mock_process_communicate.assert_called_once_with(timeout=30)
        mock_logger.info.assert_any_call("scheduler_maintenance_completed")

    def test_check_and_send_reminders_bulk_prefetch_avoids_n_plus_one(self, test_db):
        """Test that preference lookup stays bulk even with many deadlines."""
        now = datetime.now(timezone.utc)
        user_count = 40

        for user_id in range(1, user_count + 1):
            user = User(id=user_id, email=f"user{user_id}@example.com")
            test_db.add(user)
            test_db.commit()

            case = Case(
                user_id=user_id,
                case_number=f"CASE-{user_id}",
                case_type="civil",
                jurisdiction="Delhi",
                status=CaseStatus.ACTIVE,
                title=f"Title {user_id}",
            )
            test_db.add(case)
            test_db.commit()

            create_case_deadline(
                test_db,
                user_id,
                case.id,
                f"Title {user_id}",
                now + timedelta(days=30, hours=1),
                "appeal",
            )
            create_or_update_user_preference(
                test_db,
                user_id,
                f"user{user_id}@example.com",
                phone_number=f"+91{user_id:02d}00000000",
                notification_channel=NotificationChannel.SMS,
            )

        statements = []

        def capture_sql(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        event.listen(test_db.bind, "before_cursor_execute", capture_sql)
        try:
            with patch("scheduler.SessionLocal", return_value=test_db), \
                 patch("scheduler.notification_service.send_reminders") as mock_send_reminders, \
                 patch("scheduler.is_reminder_time_for_user", return_value=True):
                mock_send_reminders.return_value = [MagicMock(success=True)]
                check_and_send_reminders()
        finally:
            event.remove(test_db.bind, "before_cursor_execute", capture_sql)

        preference_queries = [
            statement for statement in statements
            if "user_preferences" in statement.lower()
        ]
        assert len(preference_queries) == 1
        assert mock_send_reminders.call_count == user_count

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

    def test_check_and_send_reminders_skips_already_sent_notifications(self, test_db):
        """Test that a previously sent reminder is not dispatched again."""
        now = datetime.now(timezone.utc)

        user = User(id=1, email="dup@example.com")
        test_db.add(user)
        test_db.commit()

        case = Case(user_id=1, case_number="CASE-dup", case_type="civil", jurisdiction="Delhi", status=CaseStatus.ACTIVE)
        test_db.add(case)
        test_db.commit()

        deadline = create_case_deadline(
            test_db,
            1,
            case.id,
            "Title",
            now + timedelta(days=30, minutes=5),
            "appeal",
        )
        pref = create_or_update_user_preference(
            test_db,
            1,
            "dup@example.com",
            phone_number="+911234567890",
            notification_channel=NotificationChannel.SMS,
        )
        log_notification(
            test_db,
            deadline_id=deadline.id,
            user_id=1,
            channel=NotificationChannel.SMS,
            recipient=pref.phone_number,
            days_before=30,
            status=NotificationStatus.SENT,
            message_id="sms-existing",
        )

        with patch("scheduler.SessionLocal", return_value=test_db), \
             patch("scheduler.notification_service.send_sms_reminder") as mock_sms_reminder, \
             patch("scheduler.notification_service.send_email_reminder") as mock_email_reminder, \
             patch("scheduler.is_reminder_time_for_user", return_value=True):
            check_and_send_reminders()

        mock_sms_reminder.assert_not_called()
        mock_email_reminder.assert_not_called()

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
