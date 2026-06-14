"""
Domain layer for deadline business logic.

Owns deadline calculation, priority classification, validation,
and overlap detection. Services consume this; they do not reimplement.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

# Priority thresholds — single source of truth
_CRITICAL_DAYS = 3
_HIGH_DAYS = 10
_MEDIUM_DAYS = 30

# Notification eligibility thresholds
_NOTIFICATION_THRESHOLDS: List[int] = [30, 10, 3, 1]


@dataclass(frozen=True)
class DeadlinePriority:
    label: str
    color: str  # hex color for UI
    urgency_label: str


class DeadlineEngine:
    """Calculate deadline properties and validate business rules."""

    @staticmethod
    def priority(days_until: int) -> DeadlinePriority:
        """Classify deadline priority by days remaining."""
        if days_until <= _CRITICAL_DAYS:
            return DeadlinePriority("critical", "#ff5252", "URGENT")
        if days_until <= _HIGH_DAYS:
            return DeadlinePriority("high", "#ff9100", "SOON")
        if days_until <= _MEDIUM_DAYS:
            return DeadlinePriority("medium", "#1a5490", "REMINDER")
        return DeadlinePriority("low", "#1a5490", "REMINDER")

    @staticmethod
    def days_until(due_date: dt.datetime, now: Optional[dt.datetime] = None) -> int:
        """Calculate calendar days until deadline. Returns 0 if past."""
        if now is None:
            now = dt.datetime.now(dt.timezone.utc)
        if due_date.tzinfo is None:
            due_date = due_date.replace(tzinfo=dt.timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt.timezone.utc)
        delta = (due_date.date() - now.date()).days
        return max(0, delta)

    @staticmethod
    def is_past(due_date: dt.datetime, now: Optional[dt.datetime] = None) -> bool:
        """Check if deadline has passed."""
        if now is None:
            now = dt.datetime.now(dt.timezone.utc)
        if due_date.tzinfo is None:
            due_date = due_date.replace(tzinfo=dt.timezone.utc)
        return due_date < now

    @staticmethod
    def validate_not_past(due_date: dt.datetime, now: Optional[dt.datetime] = None) -> bool:
        """Validate deadline is not in the past. Returns True if valid."""
        return not DeadlineEngine.is_past(due_date, now)

    @staticmethod
    def notification_thresholds() -> List[int]:
        """Return standard notification reminder thresholds."""
        return list(_NOTIFICATION_THRESHOLDS)

    @staticmethod
    def is_notification_eligible(days_until: int) -> bool:
        """Check if days_until matches a notification threshold."""
        return days_until in _NOTIFICATION_THRESHOLDS

    @staticmethod
    def first_action(deadline_type: Optional[str]) -> str:
        """Return deterministic next-action suggestion for a deadline type."""
        normalized = str(deadline_type or "other").strip().lower()
        mapping = {
            "appeal": "File appeal memo",
            "filing": "Gather filing documents",
            "submission": "Prepare and submit the required filing",
            "response": "Draft the response and review supporting records",
            "hearing": "Consult counsel and prepare the hearing bundle",
            "order": "Review the order and confirm the next step",
            "other": "Review the deadline details and plan the next step",
            "manual": "Review the deadline details and plan the next step",
        }
        return mapping.get(normalized, "Review the deadline details and plan the next step")

    @staticmethod
    def detect_overlap(
        deadlines: List[Dict[str, Any]],
        window_days: int = 3,
    ) -> List[tuple[int, int]]:
        """Detect pairs of deadlines that fall within window_days of each other."""
        overlaps = []
        sorted_dl = sorted(deadlines, key=lambda d: d.get("deadline_date", dt.datetime.min))
        for i in range(len(sorted_dl)):
            for j in range(i + 1, len(sorted_dl)):
                d1 = sorted_dl[i].get("deadline_date")
                d2 = sorted_dl[j].get("deadline_date")
                if d1 is None or d2 is None:
                    continue
                if abs((d2.date() - d1.date()).days) <= window_days:
                    overlaps.append((sorted_dl[i].get("id"), sorted_dl[j].get("id")))
        return overlaps