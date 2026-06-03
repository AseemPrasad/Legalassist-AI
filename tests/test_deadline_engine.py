from core.deadline_engine import calculate_deadline, get_deadline_first_action
from datetime import datetime


def test_exclude_weekend_basic():
    # Friday 2026-05-15 + 1 business day -> Monday 2026-05-18
    res = calculate_deadline("2026-05-15T09:00:00", business_days=1, timezone="UTC", exclude_weekends=True)
    assert res["components"]["after_business_day_add"].startswith("2026-05-18")


def test_holiday_skipped():
    # Friday 2026-05-15 + 1 business day but Monday is holiday -> Tuesday
    res = calculate_deadline(
        "2026-05-15T09:00:00",
        business_days=1,
        timezone="UTC",
        exclude_weekends=True,
        holidays=["2026-05-18"],
    )
    assert res["components"]["after_business_day_add"].startswith("2026-05-19")


def test_timezone_preserved_and_emergency_extension():
    # Use America/New_York tz
    res = calculate_deadline(
        "2026-05-14T23:00:00",
        business_days=1,
        timezone="America/New_York",
        exclude_weekends=True,
        emergency_extension_days=2,
    )
    # Final day should include the timezone offset
    assert res["deadline"].endswith("-04:00") or res["deadline"].endswith("-05:00")
    # emergency extension applied
    assert res["components"]["emergency_extension_days"] == 2


def test_get_deadline_first_action_is_deterministic():
    assert get_deadline_first_action("appeal") == "File appeal memo"
    assert get_deadline_first_action("filing") == "Gather filing documents"
    assert get_deadline_first_action("submission") == "Prepare and submit the required filing"
    assert get_deadline_first_action("response") == "Draft the response and review supporting records"
    assert get_deadline_first_action("hearing") == "Consult counsel and prepare the hearing bundle"
    assert get_deadline_first_action("order") == "Review the order and confirm the next step"
    assert get_deadline_first_action("manual") == "Review the deadline details and plan the next step"
    assert get_deadline_first_action(None) == "Review the deadline details and plan the next step"
