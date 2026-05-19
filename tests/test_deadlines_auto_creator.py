import datetime as dt
import types

import pytest
from unittest.mock import MagicMock

import services.deadlines_auto_creator as deadlines_auto_creator
from services.deadlines_auto_creator import _extract_days_from_text, _validate_days_value


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


def test_auto_create_deadlines_from_remedies_keeps_utc_across_midnight(monkeypatch):
    fixed_now = dt.datetime(2026, 5, 19, 23, 55, tzinfo=dt.timezone.utc)

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

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    deadlines_auto_creator.auto_create_deadlines_from_remedies(
        db=mock_db,
        user_id=1,
        case_id=42,
        case_title="Boundary Case",
        remedies={"appeal_days": "1", "appeal_court": "High Court"},
        document_id=99,
    )

    created_deadline = mock_db.add.call_args[0][0]
    assert created_deadline.deadline_date == fixed_now + dt.timedelta(days=1)
    assert created_deadline.deadline_date.tzinfo == dt.timezone.utc


def test_auto_create_deadlines_from_remedies_skips_only_matching_source_days(monkeypatch):
    fixed_now = dt.datetime(2026, 5, 19, 23, 55, tzinfo=dt.timezone.utc)

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

    existing_event = types.SimpleNamespace(event_metadata={"source_days": 30, "document_id": 7})
    timeline_query = MagicMock()
    timeline_query.filter.return_value.all.return_value = [existing_event]

    mock_db = MagicMock()
    mock_db.query.return_value = timeline_query

    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=MagicMock()),
    )

    deadlines_auto_creator.auto_create_deadlines_from_remedies(
        db=mock_db,
        user_id=1,
        case_id=42,
        case_title="Boundary Case",
        remedies={"appeal_days": "31", "appeal_court": "High Court"},
        document_id=8,
    )

    assert mock_db.add.call_count == 1
    created_deadline = mock_db.add.call_args[0][0]
    assert created_deadline.deadline_date == fixed_now + dt.timedelta(days=31)


def test_auto_create_deadlines_from_remedies_skips_same_source_days(monkeypatch):
    fixed_now = dt.datetime(2026, 5, 19, 23, 55, tzinfo=dt.timezone.utc)

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

    existing_event = types.SimpleNamespace(event_metadata={"source_days": 30, "document_id": 7})
    timeline_query = MagicMock()
    timeline_query.filter.return_value.all.return_value = [existing_event]

    mock_db = MagicMock()
    mock_db.query.return_value = timeline_query

    monkeypatch.setattr(
        deadlines_auto_creator,
        "timeline_service",
        types.SimpleNamespace(create_event=MagicMock()),
    )

    deadlines_auto_creator.auto_create_deadlines_from_remedies(
        db=mock_db,
        user_id=1,
        case_id=42,
        case_title="Boundary Case",
        remedies={"appeal_days": "30", "appeal_court": "High Court"},
        document_id=99,
    )

    assert mock_db.add.call_count == 0
