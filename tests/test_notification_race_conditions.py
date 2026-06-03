import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.base import Base
from db.models.cases import Case, CaseDeadline, CaseStatus
from db.models.notifications import (
    NotificationChannel,
    NotificationLog,
    NotificationStatus,
    UserPreference,
)
from db.models.auth import User
from db.crud.notifications import get_or_create_notification_log


@pytest.fixture()
def shared_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
    yield session_factory
    engine.dispose()


def test_overlapping_sms_reminders_create_one_log_and_send_once(shared_session_factory):
    setup_db = shared_session_factory()
    now = datetime.now(timezone.utc)

    user = User(id=1, email="race@example.com")
    setup_db.add(user)
    setup_db.commit()

    case = Case(
        user_id=1,
        case_number="CASE-RACE",
        case_type="civil",
        jurisdiction="Delhi",
        status=CaseStatus.ACTIVE,
        title="Race Case",
    )
    setup_db.add(case)
    setup_db.commit()
    setup_db.refresh(case)

    deadline = CaseDeadline(
        user_id=1,
        case_id=case.id,
        case_title="Race Case",
        deadline_date=now + timedelta(days=30, hours=1),
        deadline_type="appeal",
    )
    setup_db.add(deadline)
    setup_db.commit()
    setup_db.refresh(deadline)

    setup_db.close()

    def run_once():
        local_db = shared_session_factory()
        try:
            log, created = get_or_create_notification_log(
                db=local_db,
                deadline_id=deadline.id,
                user_id=1,
                channel=NotificationChannel.SMS,
                recipient="+911111111111",
                days_before=30,
            )
            local_db.commit()
            return log.id, created
        finally:
            local_db.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: run_once(), range(2)))

    verify_db = shared_session_factory()
    try:
        logs = verify_db.query(NotificationLog).filter(
            NotificationLog.deadline_id == deadline.id,
            NotificationLog.days_before == 30,
            NotificationLog.channel == NotificationChannel.SMS,
        ).all()
    finally:
        verify_db.close()

    assert len(logs) == 1
    assert sorted(created for _, created in results) == [False, True]
    assert logs[0].status in (NotificationStatus.PENDING, NotificationStatus.SENT)
