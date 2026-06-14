"""
Domain layer for notification eligibility business logic.

Owns reminder thresholds, channel selection, and send eligibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from domain.deadline import DeadlineEngine


@dataclass(frozen=True)
class ChannelOrder:
    primary: str
    fallback: str


class NotificationEligibility:
    """Determine if and when a notification should be sent."""

    @staticmethod
    def reminder_thresholds() -> List[int]:
        """Return standard reminder day thresholds."""
        return DeadlineEngine.notification_thresholds()

    @staticmethod
    def is_eligible(days_until: int) -> bool:
        """Check if days_until matches a notification threshold."""
        return DeadlineEngine.is_notification_eligible(days_until)

    @staticmethod
    def channel_order(prefers_email: bool) -> ChannelOrder:
        """Return channel priority based on user preference."""
        if prefers_email:
            return ChannelOrder("email", "sms")
        return ChannelOrder("sms", "email")

    @staticmethod
    def urgency_color(days_until: int) -> str:
        """Return UI color for urgency level."""
        priority = DeadlineEngine.priority(days_until)
        return priority.color

    @staticmethod
    def urgency_label(days_until: int) -> str:
        """Return urgency label for UI."""
        priority = DeadlineEngine.priority(days_until)
        return priority.urgency_label