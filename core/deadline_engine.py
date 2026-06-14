"""
Deadline calculation and business rules.

Delegates to domain/deadline.py for core logic. This module retains
jurisdiction-specific rules (holidays, weekends, court schedules) that are
infrastructure concerns, not pure domain logic.
"""

from datetime import datetime, time, timedelta, date
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict, Any

from domain.deadline import DeadlineEngine


def _parse_date(value: Any, tz: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError:
            dt = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    else:
        dt = dt.astimezone(ZoneInfo(tz))

    return dt


_JURISDICTION_WEEKENDS = {
    "US": {5, 6},
    "NY": {5, 6},
    "CA": {5, 6},
    "UK": {5, 6},
    "IL": {4, 5},
    "IN": {5, 6},
    "BD": {4, 5},
    "AE": {4, 5},
    "NP": {5, 6},
    "EG": {4, 5},
    "SA": {4, 5},
    "PK": {5, 6},
}

COURT_HOLIDAYS = {
    "IN_SC": [
        "2026-01-26",
        "2026-03-03",
        "2026-08-15",
        "2026-10-02",
        "2026-11-08",
        "2026-12-25",
    ],
    "IN_DHC": [
        "2026-01-26",
        "2026-03-03",
        "2026-08-15",
        "2026-10-02",
        "2026-11-08",
        "2026-12-25",
    ],
}


def _is_weekend(dt: date, jurisdiction: Optional[str] = None) -> bool:
    weekend_days = _JURISDICTION_WEEKENDS.get(jurisdiction.upper() if jurisdiction else "", {5, 6})
    return dt.weekday() in weekend_days


def calculate_deadline(
    start: Any,
    business_days: int,
    timezone: str = "UTC",
    exclude_weekends: bool = True,
    holidays: Optional[List[str]] = None,
    jurisdiction: Optional[str] = None,
    emergency_extension_days: int = 0,
    filing_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate a deadline applying business day rules, holidays, and jurisdiction rules."""
    tz = timezone or "UTC"
    dt = _parse_date(start, tz)

    holidays_set = set()
    if holidays:
        for h in holidays:
            try:
                date.fromisoformat(h)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid holiday date format: {h!r}. Expected YYYY-MM-DD.")
        holidays_set = set(holidays)

    if jurisdiction:
        for h in COURT_HOLIDAYS.get(jurisdiction.upper(), []):
            holidays_set.add(h)

    remaining = max(0, int(business_days))
    current = dt
    steps = 0

    def _roll_forward(d):
        while True:
            if exclude_weekends and _is_weekend(d.date(), jurisdiction):
                d += timedelta(days=1)
                continue
            if d.date().isoformat() in holidays_set:
                d += timedelta(days=1)
                continue
            break
        return d

    while remaining > 0:
        current += timedelta(days=1)
        steps += 1
        current = _roll_forward(current)
        remaining -= 1

    if int(business_days) == 0:
        current = _roll_forward(current)

    adjusted_for_weekends_holidays = current

    jurisdiction_adjustment = 0
    if jurisdiction:
        rules = {
            "NY": {"cutoff_hour": 17, "add_days_after_cutoff": 1},
            "CA": {"cutoff_hour": 16, "add_days_after_cutoff": 1},
        }
        r = rules.get(jurisdiction.upper())
        if r and filing_time:
            try:
                fh = int(filing_time.split(":")[0])
                if fh >= r.get("cutoff_hour", 24):
                    jurisdiction_adjustment = r.get("add_days_after_cutoff", 0)
            except Exception:
                jurisdiction_adjustment = 0

    final = adjusted_for_weekends_holidays
    if jurisdiction_adjustment:
        final += timedelta(days=jurisdiction_adjustment)
        final = _roll_forward(final)
    if emergency_extension_days:
        final += timedelta(days=int(emergency_extension_days))
        final = _roll_forward(final)

    while (exclude_weekends and _is_weekend(final.date(), jurisdiction)) or (final.date().isoformat() in holidays_set):
        final = final + timedelta(days=1)

    return {
        "deadline": final.isoformat(),
        "components": {
            "start": dt.isoformat(),
            "after_business_day_add": adjusted_for_weekends_holidays.isoformat(),
            "jurisdiction_adjustment_days": jurisdiction_adjustment,
            "emergency_extension_days": int(emergency_extension_days),
            "timezone": tz,
            "steps_taken": steps,
        },
    }


__all__ = ["calculate_deadline"]


def get_deadline_first_action(deadline_type: Optional[str]) -> str:
    """Return a short deterministic next-action suggestion for a deadline type."""
    return DeadlineEngine.first_action(deadline_type)


__all__.append("get_deadline_first_action")