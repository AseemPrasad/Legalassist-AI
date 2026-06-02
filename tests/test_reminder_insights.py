from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from analytics_engine import ReminderInsightsEngine
from db.base import Base
from db.models import Case, CaseDeadline, CaseTimeline, NotificationChannel, NotificationLog, NotificationStatus, User


@pytest.fixture()
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def _create_case(db, user_id: int, case_number: str, jurisdiction: str, title: str, court_name: str) -> Case:
    case = Case(
        user_id=user_id,
        case_number=case_number,
        case_type="civil",
        jurisdiction=jurisdiction,
        title=title,
    )
    db.add(case)
    db.flush()
    return case


def _create_deadline(
    db,
    user_id: int,
    case: Case,
    deadline_type: str,
    court_name: str,
    deadline_date: datetime,
    is_completed: bool,
) -> CaseDeadline:
    deadline = CaseDeadline(
        user_id=user_id,
        case_id=case.id,
        case_title=case.title or case.case_number,
        court_name=court_name,
        deadline_date=deadline_date,
        deadline_type=deadline_type,
        first_action="Review the deadline details and plan the next step",
        is_completed=is_completed,
        status="completed" if is_completed else "active",
    )
    db.add(deadline)
    db.flush()
    return deadline


class TestReminderInsightsEngine:
    def test_build_insights_uses_last_touch_and_groups_by_dimensions(self, test_db):
        user = User(email="insights@example.com")
        test_db.add(user)
        test_db.flush()

        now = datetime.now(timezone.utc).replace(microsecond=0)

        delhi_case = _create_case(test_db, user.id, "CASE-DELHI", "Delhi", "Delhi matter", "High Court")
        mumbai_case = _create_case(test_db, user.id, "CASE-MUM", "Mumbai", "Mumbai matter", "District Court")

        effective_deadline = _create_deadline(
            test_db,
            user.id,
            delhi_case,
            "appeal",
            "High Court",
            now - timedelta(days=1),
            True,
        )
        effective_sent_early = now - timedelta(days=10)
        effective_sent_latest = now - timedelta(days=9)
        earlier_log = NotificationLog(
            deadline_id=effective_deadline.id,
            user_id=user.id,
            channel=NotificationChannel.SMS,
            status=NotificationStatus.SENT,
            recipient="+10000000000",
            days_before=10,
            sent_at=effective_sent_early,
        )
        latest_log = NotificationLog(
            deadline_id=effective_deadline.id,
            user_id=user.id,
            channel=NotificationChannel.SMS,
            status=NotificationStatus.DELIVERED,
            recipient="+10000000000",
            days_before=9,
            sent_at=effective_sent_latest,
            delivered_at=effective_sent_latest + timedelta(hours=2),
        )
        test_db.add_all([earlier_log, latest_log])
        test_db.flush()

        test_db.add(
            CaseTimeline(
                case_id=delhi_case.id,
                event_type="deadline_completed",
                description="Marked appeal deadline as completed",
                event_date=effective_sent_latest + timedelta(days=2),
                event_metadata={"deadline_id": effective_deadline.id},
            )
        )

        dropoff_deadline = _create_deadline(
            test_db,
            user.id,
            mumbai_case,
            "response",
            "District Court",
            now - timedelta(days=3),
            False,
        )
        dropoff_log = NotificationLog(
            deadline_id=dropoff_deadline.id,
            user_id=user.id,
            channel=NotificationChannel.EMAIL,
            status=NotificationStatus.SENT,
            recipient="user@example.com",
            days_before=7,
            sent_at=now - timedelta(days=8),
        )
        test_db.add(dropoff_log)
        test_db.commit()

        insights = ReminderInsightsEngine.build_insights(test_db, attribution_window_days=14, user_id=user.id)

        summary = insights["summary"]
        frame = insights["frame"]
        by_jurisdiction = insights["by_jurisdiction"]
        by_court = insights["by_court"]
        by_deadline_type = insights["by_deadline_type"]
        by_channel = insights["by_channel"]

        assert summary["reminder_count"] == 2
        assert summary["effective_reminders"] == 1
        assert summary["drop_off_reminders"] == 1
        assert summary["effectiveness_rate"] == 50.0
        assert len(frame) == 2

        effective_row = frame.loc[frame["deadline_id"] == effective_deadline.id].iloc[0]
        assert effective_row["notification_log_id"] == latest_log.id
        assert bool(effective_row["effective"]) is True
        assert bool(effective_row["drop_off"]) is False

        dropoff_row = frame.loc[frame["deadline_id"] == dropoff_deadline.id].iloc[0]
        assert bool(dropoff_row["effective"]) is False
        assert bool(dropoff_row["drop_off"]) is True

        assert set(by_jurisdiction["jurisdiction"]) == {"Delhi", "Mumbai"}
        assert set(by_court["court_name"]) == {"High Court", "District Court"}
        assert set(by_deadline_type["deadline_type"]) == {"appeal", "response"}
        assert set(by_channel["channel"]) == {"sms", "email"}
        assert int(by_channel.loc[by_channel["channel"] == "sms", "effective_reminders"].iloc[0]) == 1
        assert int(by_channel.loc[by_channel["channel"] == "email", "drop_off_reminders"].iloc[0]) == 1
