"""Reminder decision logic extracted for easy testing.

Pure helper functions and an orchestration builder that converts upcoming
deadlines into actionable reminder jobs (deadline, user_pref, days_left).
"""
from typing import Iterable, Tuple
import datetime as dt
import pytz
from sqlalchemy.orm import Session

from db.models.cases import CaseDeadline
from db.models.notifications import UserPreference


def should_process_threshold(days_left: int) -> bool:
    """Return True when the days_left value matches configured thresholds."""
    return days_left in (30, 10, 3, 1)


def is_notify_enabled(days_left: int, user_preference: UserPreference) -> bool:
    """Return True if the user has enabled reminders for this threshold."""
    if user_preference is None:
        return False
    if days_left == 30:
        return bool(user_preference.notify_30_days)
    if days_left == 10:
        return bool(user_preference.notify_10_days)
    if days_left == 3:
        return bool(user_preference.notify_3_days)
    if days_left == 1:
        return bool(user_preference.notify_1_day)
    return False


def is_reminder_time_for_user(user_timezone: str, reminder_hour: int = 8) -> bool:
    """Return True when current hour in user's timezone equals `reminder_hour`.

    Falls back to UTC when timezone is invalid.
    """
    try:
        if not user_timezone or not isinstance(user_timezone, str):
            raise ValueError("Invalid timezone type")
        tz = pytz.timezone(user_timezone)
        user_now = dt.datetime.now(tz)
        return user_now.hour == reminder_hour
    except (pytz.exceptions.UnknownTimeZoneError, ValueError, AttributeError):
        # Fallback to UTC if timezone is invalid
        user_now = dt.datetime.now(dt.timezone.utc)
        return user_now.hour == reminder_hour


def build_reminder_jobs(upcoming_deadlines: Iterable[CaseDeadline], db: Session) -> Iterable[Tuple[CaseDeadline, UserPreference, int]]:
    """Yield (deadline, user_pref, days_left) for deadlines that should be processed.

    This function centralizes threshold checks, preference lookup and timezone
    eligibility so the scheduler can simply iterate and dispatch jobs.
    """

    user_ids = {deadline.user_id for deadline in upcoming_deadlines}
    user_prefs = {pref.user_id: pref for pref in db.query(UserPreference).filter(UserPreferance.user_id.in(user_ids)).all()
        }

    

    for deadline in upcoming_deadlines:
        days_left = deadline.days_until_deadline()
        if not should_process_threshold(days_left):
            continue

        user_pref = user_prefs.get(deadline.user_id)
    
        if not user_pref:
            continue

        if not is_notify_enabled(days_left, user_pref):
            continue

        if not is_reminder_time_for_user(user_pref.timezone):
            continue

        yield (deadline, user_pref, days_left)
