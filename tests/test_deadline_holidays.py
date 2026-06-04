from datetime import datetime
from core.deadline_engine import calculate_deadline, COURT_HOLIDAYS


def test_calculate_deadline_with_court_holidays():
    # Republic day 2026-01-26 is a Monday.
    # Start on Friday 2026-01-23. Add 1 business day.
    # Friday -> Sat (skip) -> Sun (skip) -> Mon (Republic Day holiday, skip) -> Tuesday 2026-01-27
    res = calculate_deadline(
        start="2026-01-23T10:00:00",
        business_days=1,
        jurisdiction="IN_SC",
    )
    
    # Tuesday 27th Jan 2026
    assert "2026-01-27" in res["deadline"]


def test_calculate_deadline_same_day_weekend_push():
    # Start on Saturday 2026-01-24. Add 0 business days.
    # Output should get pushed to Monday 2026-01-26, but that is Republic Day.
    # So it should get pushed to Tuesday 2026-01-27.
    res = calculate_deadline(
        start="2026-01-24T10:00:00",
        business_days=0,
        jurisdiction="IN_SC",
    )
    
    assert "2026-01-27" in res["deadline"]
