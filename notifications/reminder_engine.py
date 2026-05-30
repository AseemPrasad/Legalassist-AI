"""Reminder decision logic extracted for easy testing.

Pure helper functions and an orchestration planner that converts upcoming
deadlines plus prefetched user preferences into actionable reminder candidates.
"""
import datetime as dt
from typing import Callable, Dict, Iterable, List, Tuple

import pytz

from db.crud.notifications import get_prefs_by_user_ids
from db.models.cases import CaseDeadline
from db.models.notifications import UserPreference


def should_process_threshold(days_left: int) -> bool:
    """Return True when the days_left value matches configured thresholds.
    Note: For dynamic user-configurable thresholds, this now acts as a broad
    check but is overridden by the user's specific thresholds during dispatch.
    """
    return True


def is_notify_enabled(days_left: int, user_preference: UserPreference) -> bool:
    """Return True if the user has enabled reminders for this threshold."""
    if user_preference is None:
        return False
    if hasattr(user_preference, "get_reminder_thresholds"):
        thresholds = user_preference.get_reminder_thresholds()
        return days_left in thresholds

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


ReminderDispatchCandidate = Tuple[CaseDeadline, int, UserPreference]


def _ensure_utc_datetime(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _calculate_days_left(deadline: CaseDeadline, now_utc: dt.datetime) -> int:
    deadline_date = deadline.deadline_date
    if deadline_date is None:
        return 0
    if deadline_date.tzinfo is None:
        deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)
    else:
        deadline_date = deadline_date.astimezone(dt.timezone.utc)
    return max(0, (deadline_date.date() - now_utc.date()).days)


def _build_reminder_dispatch_candidates(
    deadlines: Iterable[CaseDeadline],
    prefs_by_user_id: Dict[int, UserPreference],
    now_utc: dt.datetime,
    reminder_time_checker: Callable[[str], bool],
) -> List[ReminderDispatchCandidate]:
    candidates: List[ReminderDispatchCandidate] = []
    normalized_now_utc = _ensure_utc_datetime(now_utc)

    for deadline in deadlines:
        days_left = _calculate_days_left(deadline, normalized_now_utc)
        if not should_process_threshold(days_left):
            continue

        user_pref = prefs_by_user_id.get(deadline.user_id)
        if not user_pref:
            continue

        if not is_notify_enabled(days_left, user_pref):
            continue

        if not reminder_time_checker(user_pref.timezone):
            continue

        candidates.append((deadline, days_left, user_pref))

    return candidates


def get_reminder_dispatch_candidates(
    db,
    days_before: int,
    now_utc: dt.datetime,
    reminder_time_checker: Callable[[str], bool] | None = None,
) -> List[ReminderDispatchCandidate]:
    """Build reminder candidates from the database and current UTC time."""
    if reminder_time_checker is None:
        reminder_time_checker = is_reminder_time_for_user

    now_utc = _ensure_utc_datetime(now_utc)
    target_utc = (now_utc + dt.timedelta(days=days_before)).replace(
        hour=23,
        minute=59,
        second=59,
        microsecond=999999,
    )
    deadlines = db.query(CaseDeadline).filter(
        CaseDeadline.status == "active",
        CaseDeadline.deadline_date <= target_utc,
        CaseDeadline.deadline_date > now_utc,
    ).all()

    prefs_by_user_id: Dict[int, UserPreference] = {}
    if deadlines:
        prefs = get_prefs_by_user_ids(db, {deadline.user_id for deadline in deadlines})
        prefs_by_user_id = {pref.user_id: pref for pref in prefs}

    return _build_reminder_dispatch_candidates(
        deadlines,
        prefs_by_user_id,
        now_utc,
        reminder_time_checker,
    )


def plan_eligible_reminders(
    deadlines: Iterable[CaseDeadline],
    prefs_by_user_id: Dict[int, UserPreference],
    now_utc: dt.datetime | None = None,
    reminder_time_checker: Callable[[str], bool] | None = None,
) -> List[ReminderDispatchCandidate]:
    """Plan reminder candidates from in-memory deadlines and preferences."""
    if reminder_time_checker is None:
        reminder_time_checker = is_reminder_time_for_user

    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)

    return _build_reminder_dispatch_candidates(
        deadlines,
        prefs_by_user_id,
        now_utc,
        reminder_time_checker,
    )
