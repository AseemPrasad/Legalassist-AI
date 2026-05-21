"""Reminder decision logic extracted for easy testing.

Pure helper functions and an orchestration planner that converts upcoming
deadlines plus prefetched user preferences into actionable reminder candidates.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List
import datetime as dt
import pytz

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


@dataclass(frozen=True)
class ReminderCandidate:
    deadline: CaseDeadline
    days_left: int
    timezone: str
    notify_30_days: bool
    notify_10_days: bool
    notify_3_days: bool
    notify_1_day: bool
    notification_channel: object
    user_preference: UserPreference


def plan_eligible_reminders(
    deadlines: Iterable[CaseDeadline],
    prefs_by_user_id: Dict[int, UserPreference],
    reminder_time_checker: Callable[[str], bool] = is_reminder_time_for_user,
) -> List[ReminderCandidate]:
    """Plan reminder candidates from in-memory deadlines and preferences."""
    candidates: List[ReminderCandidate] = []

    for deadline in deadlines:
        days_left = deadline.days_until_deadline()
        if not should_process_threshold(days_left):
            continue

        user_pref = prefs_by_user_id.get(deadline.user_id)
        if not user_pref:
            continue

        if not is_notify_enabled(days_left, user_pref):
            continue

        if not reminder_time_checker(user_pref.timezone):
            continue

        candidates.append(
            ReminderCandidate(
                deadline=deadline,
                days_left=days_left,
                timezone=user_pref.timezone or "UTC",
                notify_30_days=bool(user_pref.notify_30_days),
                notify_10_days=bool(user_pref.notify_10_days),
                notify_3_days=bool(user_pref.notify_3_days),
                notify_1_day=bool(user_pref.notify_1_day),
                notification_channel=user_pref.notification_channel,
                user_preference=user_pref,
            )
        )

    return candidates
