"""
Centralized timezone-aware datetime utility.

Replaces mixed naive/aware datetime construction across the codebase.
All database writes store UTC. All business logic operates on aware objects.
The Clock abstraction allows freezing and travel for testing.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Callable
from contextlib import contextmanager


class Clock:
    """
    Centralized timezone-aware datetime factory.

    Production code uses the real clock. Tests can freeze() or travel()
    to control time without mocking datetime directly.

    All returned datetimes are timezone-aware (UTC).
    """

    _frozen_at: Optional[dt.datetime] = None
    _offset: Optional[dt.timedelta] = None

    @classmethod
    def now(cls) -> dt.datetime:
        """Return current time as timezone-aware UTC datetime."""
        if cls._frozen_at is not None:
            return cls._frozen_at
        base = dt.datetime.now(dt.timezone.utc)
        if cls._offset is not None:
            return base + cls._offset
        return base

    @classmethod
    def utc(cls) -> dt.datetime:
        """Alias for now(). Returns UTC-aware datetime."""
        return cls.now()

    @classmethod
    def parse(cls, iso_string: str) -> dt.datetime:
        """Parse ISO 8601 string to timezone-aware datetime."""
        parsed = dt.datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed

    @classmethod
    def isoformat(cls, when: Optional[dt.datetime] = None) -> str:
        """Return ISO 8601 string with explicit Z suffix."""
        target = when or cls.now()
        if target.tzinfo is None:
            target = target.replace(tzinfo=dt.timezone.utc)
        return target.strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    @contextmanager
    def freeze(cls, frozen_time: dt.datetime):
        """
        Freeze time for testing. Yields control, then restores.
        
        Usage:
            with Clock.freeze(datetime(2024, 1, 1, tzinfo=timezone.utc)):
                assert Clock.now() == frozen_time
        """
        previous = cls._frozen_at
        cls._frozen_at = frozen_time
        try:
            yield
        finally:
            cls._frozen_at = previous

    @classmethod
    @contextmanager
    def travel(cls, offset: dt.timedelta):
        """
        Shift time by offset for testing. Yields control, then restores.
        
        Usage:
            with Clock.travel(timedelta(days=1)):
                assert Clock.now() > real_now
        """
        previous = cls._offset
        cls._offset = offset
        try:
            yield
        finally:
            cls._offset = previous

    @classmethod
    def reset(cls) -> None:
        """Clear any freeze or travel state. Return to real time."""
        cls._frozen_at = None
        cls._offset = None


def _utc_datetime(value: dt.datetime) -> dt.datetime:
    """
    Ensure a datetime is UTC-aware. Idempotent.
    Replaces scattered _to_utc_datetime helpers.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)