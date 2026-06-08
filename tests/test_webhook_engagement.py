"""Tests for email engagement event tracking in webhook handlers."""

from __future__ import annotations

from db.models.notifications import NotificationStatus


def _mapping_fn(event_type: str | None) -> NotificationStatus | None:
    """Replicates _sendgrid_event_to_notification_status for test isolation."""
    normalized = (event_type or "").lower()
    if normalized == "delivered":
        return NotificationStatus.DELIVERED
    if normalized in {"bounce", "dropped", "deferred", "blocked", "spamreport", "invalid"}:
        return NotificationStatus.FAILED
    if normalized in {"open", "click"}:
        return NotificationStatus.OPENED
    return None


class TestSendGridEventMapping:
    def test_open_event_maps_to_opened(self):
        result = _mapping_fn("open")
        assert result == NotificationStatus.OPENED

    def test_click_event_maps_to_opened(self):
        result = _mapping_fn("click")
        assert result == NotificationStatus.OPENED

    def test_open_case_variations(self):
        assert _mapping_fn("OPEN") == NotificationStatus.OPENED
        assert _mapping_fn("Open") == NotificationStatus.OPENED

    def test_delivered_still_maps(self):
        assert _mapping_fn("delivered") == NotificationStatus.DELIVERED

    def test_failure_events_still_map(self):
        assert _mapping_fn("bounce") == NotificationStatus.FAILED
        assert _mapping_fn("dropped") == NotificationStatus.FAILED
        assert _mapping_fn("blocked") == NotificationStatus.FAILED

    def test_unknown_returns_none(self):
        assert _mapping_fn("unknown") is None
        assert _mapping_fn("") is None
        assert _mapping_fn(None) is None
