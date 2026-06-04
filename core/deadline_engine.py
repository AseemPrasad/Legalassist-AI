from datetime import datetime, time, timedelta, date
from zoneinfo import ZoneInfo
from typing import List, Optional, Dict, Any


def _parse_date(value: Any, tz: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError:
            # Fall back to space-separated SQLite datetimes for Python 3.10
            dt = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    else:
        # convert to requested tz
        dt = dt.astimezone(ZoneInfo(tz))

    return dt


_JURISDICTION_WEEKENDS = {
    # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
    "US": {5, 6},       # Sat–Sun
    "NY": {5, 6},
    "CA": {5, 6},
    "UK": {5, 6},
    "IL": {4, 5},       # Fri–Sat (Israel)
    "IN": {5, 6},       # Sat–Sun (India)
    "BD": {4, 5},       # Fri–Sat (Bangladesh)
    "AE": {4, 5},       # Fri–Sat (UAE)
    "NP": {5, 6},       # Sat–Sun (Nepal)
    "EG": {4, 5},       # Fri–Sat (Egypt)
    "SA": {4, 5},       # Fri–Sat (Saudi Arabia)
    "PK": {5, 6},       # Sat–Sun (Pakistan)
}

COURT_HOLIDAYS = {
    "IN_SC": [
        "2026-01-26",  # Republic Day
        "2026-03-03",  # Holi
        "2026-08-15",  # Independence Day
        "2026-10-02",  # Gandhi Jayanti
        "2026-11-08",  # Diwali
        "2026-12-25",  # Christmas
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
    # Check jurisdiction-specific weekend days mapped in dictionary
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
    """Calculate a deadline applying business day rules, holidays, and jurisdiction rules.

    - `start`: ISO datetime string or datetime/date
    - `business_days`: number of business days to add
    - `timezone`: IANA tz name
    - `holidays`: list of ISO date strings (YYYY-MM-DD)
    - `jurisdiction`: optional key for jurisdiction-specific rules (POC)
    - `emergency_extension_days`: extra days added for emergency relief
    - `filing_time`: optional HH:MM to check against jurisdiction cutoff
    """
    tz = timezone or "UTC"
    dt = _parse_date(start, tz)

    holidays_set = set()
    if holidays:
        for h in holidays:
            try:
                # Confirm holiday date matches ISO YYYY-MM-DD pattern
                date.fromisoformat(h)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid holiday date format: {h!r}. Expected YYYY-MM-DD."
                )
        holidays_set = set(holidays)

    if jurisdiction:
        for h in COURT_HOLIDAYS.get(jurisdiction.upper(), []):
            holidays_set.add(h)

    remaining = max(0, int(business_days))
    current = dt
    steps = 0

    def _roll_forward(d):
        """Normalize *d* to the next non-weekend, non-holiday date."""
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

    # Normalize even when business_days==0 (e.g. start is a weekend/holiday)
    if int(business_days) == 0:
        # Perform rolling forward adjustment for weekends and holidays
        current = _roll_forward(current)

    adjusted_for_weekends_holidays = current

    # Apply jurisdiction-specific rules (POC)
    jurisdiction_adjustment = 0
    if jurisdiction:
        # Example rule: if filing after 17:00 local time, add 1 day in some jurisdictions
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

    # Apply adjustments sequentially so each step normalizes independently.
    final = adjusted_for_weekends_holidays
    if jurisdiction_adjustment:
        # Sequentially apply the filing cutoff adjustments first
        final += timedelta(days=jurisdiction_adjustment)
        final = _roll_forward(final)
    if emergency_extension_days:
        final += timedelta(days=int(emergency_extension_days))
        final = _roll_forward(final)

    # Recheck if the final date with adjustments lands on a holiday/weekend
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


_DEADLINE_TYPE_FIRST_ACTIONS: Dict[str, str] = {
    "appeal": "File appeal memo",
    "filing": "Gather filing documents",
    "submission": "Prepare and submit the required filing",
    "response": "Draft the response and review supporting records",
    "hearing": "Consult counsel and prepare the hearing bundle",
    "order": "Review the order and confirm the next step",
    "other": "Review the deadline details and plan the next step",
    "manual": "Review the deadline details and plan the next step",
}


def get_deadline_first_action(deadline_type: Optional[str]) -> str:
    """Return a short deterministic next-action suggestion for a deadline type."""
    normalized = str(deadline_type or "other").strip().lower()
    return _DEADLINE_TYPE_FIRST_ACTIONS.get(normalized, "Review the deadline details and plan the next step")


__all__.append("get_deadline_first_action")
