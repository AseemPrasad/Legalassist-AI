import datetime as dt
import types

import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import services.deadlines_auto_creator as deadlines_auto_creator
from database import Base, Case, CaseDeadline, CaseTimeline
from db.models import User
from services.deadlines_auto_creator import _extract_days_from_text, _validate_days_value


@pytest.fixture(scope="function")
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()


def _seed_case_context(db, user_id=1, case_id=42):
    user = User(id=user_id, email=f"user{user_id}@example.com")
    case = Case(
        id=case_id,
        user_id=user_id,
        case_number=f"CASE-{case_id}",
        case_type="civil",
        jurisdiction="Delhi",
        title="Boundary Case",
    )
    db.add_all([user, case])
    db.commit()


def _freeze_deadline_time(monkeypatch, fixed_now):
    class FixedDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(
        deadlines_auto_creator,
        "dt",
        types.SimpleNamespace(
            datetime=FixedDateTime,
            timedelta=dt.timedelta,
            timezone=dt.timezone,
        ),
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("appeal within 15 days", 15),
        ("file appeal in 7 days", 7),
        ("notice of appeal within 30 days", 30),
        ("30 days to file appeal", 30),
        ("challenge within 21 days", 21),
        ("Cost is 500 Rs, appeal in 30 days", 30),
        ("appeal within 30 day.", 30),
        ("file appeal within 30 business days", 30),
        ("appeal within 21 calendar days", 21),
        ("appeal in about 7 days", 7),
        ("notice of appeal within 15, days", 15),
        ("appeal in 30 days)", 30),
        ("appeal in\n30 days", 30),
        ("appeal: 30 days", 30),
    ],
)
def test_extract_days_from_text_variants(text, expected):
    assert _extract_days_from_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "Invalid text",
        "appeal by tomorrow",
        "30 days",
        "30days",
        "30 Days",
        "within 21 days of service",
        "payment due in 30 days",
        "file payment in 30 days",
        "30 business days",
        "21 calendar days",
        "in about 7 days",
    ],
)
def test_extract_days_from_text_invalid_inputs(text):
    assert _extract_days_from_text(text) is None


@pytest.mark.parametrize(
    "days, expected",
    [(1, True), (365, True), (0, False), (366, False)],
)
def test_validate_days_value_bounds(days, expected):
    assert _validate_days_value(days) is expected


def test_auto_create_deadlines_from_remedies_creates_deadline_and_timeline_event(monkeypatch, test_db):
    fixed_now = dt.datetime(2026, 5, 19, 23, 55, tzinfo=dt.timezone.utc)
    _freeze_deadline_time(monkeypatch, fixed_now)
    _seed_case_context(test_db, user_id=1, case_id=42)

    mock_create_event = MagicMock()
    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=mock_create_event),
    )

    deadlines_auto_creator.auto_create_deadlines_from_remedies(
        db=test_db,
        user_id=1,
        case_id=42,
        case_title="Boundary Case",
        remedies={"appeal_days": "1", "appeal_court": "High Court"},
        document_id=99,
    )

    created_deadline = test_db.query(CaseDeadline).one()
    expected_deadline = (fixed_now + dt.timedelta(days=1)).replace(tzinfo=None)
    assert created_deadline.deadline_date == expected_deadline
    assert created_deadline.deadline_type == "appeal"
    assert created_deadline.description == "Appeal deadline - High Court"

    mock_create_event.assert_called_once()
    assert mock_create_event.call_args.kwargs["db"] is test_db
    assert mock_create_event.call_args.kwargs["case_id"] == 42
    assert mock_create_event.call_args.kwargs["event_type"] == "deadline_created"
    assert mock_create_event.call_args.kwargs["metadata"] == {
        "deadline_id": created_deadline.id,
        "document_id": 99,
        "source_days": 1,
        "original_text": "1",
    }


@pytest.mark.parametrize(
    "existing_metadata, remedies, document_id",
    [
        ({"source_days": 30, "document_id": 7}, {"appeal_days": "30", "appeal_court": "High Court"}, 99),
        ({"document_id": 7}, {"appeal_days": "15", "appeal_court": "High Court"}, 7),
    ],
)
def test_auto_create_deadlines_from_remedies_dedupes_existing_timeline_event(
    monkeypatch,
    test_db,
    existing_metadata,
    remedies,
    document_id,
):
    _seed_case_context(test_db, user_id=1, case_id=42)
    test_db.add(
        CaseTimeline(
            case_id=42,
            event_type="deadline_created",
            description="Existing appeal deadline",
            event_metadata=existing_metadata,
        )
    )
    test_db.commit()

    mock_create_event = MagicMock()
    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=mock_create_event),
    )

    deadlines_auto_creator.auto_create_deadlines_from_remedies(
        db=test_db,
        user_id=1,
        case_id=42,
        case_title="Boundary Case",
        remedies=remedies,
        document_id=document_id,
    )

    assert test_db.query(CaseDeadline).count() == 0
    mock_create_event.assert_called_once()
    assert mock_create_event.call_args.kwargs["event_type"] == "deadline_skipped"
    assert mock_create_event.call_args.kwargs["metadata"]["reason"] == "matching_deadline_exists"


def test_auto_create_deadlines_from_remedies_logs_when_appeal_days_missing(caplog):
    mock_db = MagicMock()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=MagicMock()),
    )

    with caplog.at_level("WARNING"):
        deadlines_auto_creator.auto_create_deadlines_from_remedies(
            db=mock_db,
            user_id=1,
            case_id=42,
            case_title="Boundary Case",
            remedies={"appeal_court": "High Court"},
            document_id=99,
        )

    assert mock_db.add.call_count == 0
    assert "appeal_days is missing" in caplog.text
    monkeypatch.undo()


def test_auto_create_deadlines_from_remedies_logs_when_remedies_payload_invalid(caplog):
    mock_db = MagicMock()

    with caplog.at_level("WARNING"):
        deadlines_auto_creator.auto_create_deadlines_from_remedies(
            db=mock_db,
            user_id=1,
            case_id=42,
            case_title="Boundary Case",
            remedies='{"appeal_days": "30", "appeal_court": "High Court"}',
            document_id=99,
        )

    assert mock_db.add.call_count == 0
    assert "must be a mapping" in caplog.text


def test_auto_create_deadlines_from_remedies_logs_when_appeal_days_type_is_invalid(caplog):
    mock_db = MagicMock()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=MagicMock()),
    )

    with caplog.at_level("WARNING"):
        deadlines_auto_creator.auto_create_deadlines_from_remedies(
            db=mock_db,
            user_id=1,
            case_id=42,
            case_title="Boundary Case",
            remedies={"appeal_days": ["30"], "appeal_court": "High Court"},
            document_id=99,
        )

    assert mock_db.add.call_count == 0
    assert "invalid remedies payload shape" in caplog.text
    monkeypatch.undo()


def test_auto_create_deadlines_from_remedies_logs_and_emits_skip_event_for_invalid_days(caplog):
    mock_db = MagicMock()
    mock_event_creator = MagicMock()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=mock_event_creator),
    )

    with caplog.at_level("WARNING"):
        deadlines_auto_creator.auto_create_deadlines_from_remedies(
            db=mock_db,
            user_id=1,
            case_id=42,
            case_title="Boundary Case",
            remedies={"appeal_days": "tomorrow", "appeal_court": "High Court"},
            document_id=99,
        )

    assert mock_db.add.call_count == 0
    assert "appeal_days_invalid" in caplog.text
    mock_event_creator.assert_called_once()
    assert mock_event_creator.call_args.kwargs["event_type"] == "deadline_skipped"
    assert mock_event_creator.call_args.kwargs["metadata"]["reason"] == "appeal_days_invalid"
    monkeypatch.undo()


@pytest.mark.parametrize("appeal_days", ["tomorrow", "0", "366"])
def test_auto_create_deadlines_from_remedies_rejects_invalid_appeal_days(monkeypatch, test_db, appeal_days):
    _seed_case_context(test_db, user_id=1, case_id=42)

    mock_create_event = MagicMock()
    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=mock_create_event),
    )

    deadlines_auto_creator.auto_create_deadlines_from_remedies(
        db=test_db,
        user_id=1,
        case_id=42,
        case_title="Boundary Case",
        remedies={"appeal_days": appeal_days, "appeal_court": "High Court"},
        document_id=99,
    )

    assert test_db.query(CaseDeadline).count() == 0
    mock_create_event.assert_called_once()
    assert mock_create_event.call_args.kwargs["event_type"] == "deadline_skipped"
    assert mock_create_event.call_args.kwargs["metadata"]["reason"] == "appeal_days_invalid"
