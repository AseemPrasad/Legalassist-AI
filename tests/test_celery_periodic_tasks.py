import pytest
from datetime import datetime, timezone, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import (
    Base,
    Case,
    CaseStatus,
    User,
    NotificationChannel,
    create_case_deadline,
    create_or_update_user_preference,
)
from notification_service import NotificationResult
import celery_app
import scheduler


@pytest.fixture(scope="function")
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


def test_send_deadline_reminders_uses_real_scheduler_query(test_db, monkeypatch):
    now = datetime.now(timezone.utc)

    user = User(id=101, email="deadline@example.com")
    test_db.add(user)
    test_db.commit()

    case = Case(
        user_id=101,
        case_number="CASE-101",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Reminder Case",
    )
    test_db.add(case)
    test_db.commit()

    create_case_deadline(
        test_db,
        101,
        case.id,
        "Reminder Case",
        now + timedelta(days=30, hours=1),
        "appeal",
    )

    create_or_update_user_preference(
        test_db,
        101,
        "deadline@example.com",
        phone_number="+911111111111",
        notification_channel=NotificationChannel.BOTH,
    )

    monkeypatch.setattr(scheduler, "SessionLocal", lambda: test_db)
    monkeypatch.setattr(scheduler, "is_reminder_time_for_user", lambda timezone_name: True)
    monkeypatch.setattr(
        scheduler.notification_service,
        "send_reminders",
        lambda db, deadline_obj, user_preference, days_left: [
            NotificationResult(
                success=True,
                channel=NotificationChannel.SMS,
                recipient=user_preference.phone_number,
                message_id="sms-1",
                error=None,
            )
        ],
    )
    monkeypatch.setattr(scheduler, "init_db", lambda: None)

    result = celery_app.send_deadline_reminders()

    assert result["status"] == "completed"
    assert result["reminders_sent"] == 1
