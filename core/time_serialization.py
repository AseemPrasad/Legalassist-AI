"""Helpers for serializing timeline timestamps."""

from __future__ import annotations

from datetime import datetime, timezone


def to_utc_iso(value: datetime) -> str:
    """Normalize a datetime to UTC and return an ISO 8601 string."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()