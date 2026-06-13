"""
Domain layer for case lifecycle business logic.

Owns status transitions, valid states, and transition rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set


class CaseStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    APPEALED = "appealed"
    SETTLED = "settled"
    DISMISSED = "dismissed"
    CLOSED = "closed"


class CaseLifecycle:
    """Validates and executes case status transitions."""

    # Valid transitions: current_status -> set of allowed target statuses
    _TRANSITIONS: Dict[str, Set[str]] = {
        CaseStatus.PENDING: {CaseStatus.ACTIVE, CaseStatus.DISMISSED},
        CaseStatus.ACTIVE: {CaseStatus.SETTLED, CaseStatus.DISMISSED, CaseStatus.APPEALED, CaseStatus.CLOSED},
        CaseStatus.APPEALED: {CaseStatus.ACTIVE, CaseStatus.DISMISSED, CaseStatus.CLOSED},
        CaseStatus.SETTLED: set(),
        CaseStatus.DISMISSED: set(),
        CaseStatus.CLOSED: set(),
    }

    @classmethod
    def can_transition(cls, current: str, target: str) -> bool:
        """Check if transition from current to target is valid."""
        return target in cls._TRANSITIONS.get(current, set())

    @classmethod
    def valid_targets(cls, current: str) -> List[str]:
        """Return list of valid target statuses from current."""
        return sorted(cls._TRANSITIONS.get(current, set()))

    @classmethod
    def validate_transition(cls, current: str, target: str) -> None:
        """Raise ValueError if transition is invalid."""
        if not cls.can_transition(current, target):
            raise ValueError(
                f"Invalid transition: {current} -> {target}. "
                f"Valid targets: {cls.valid_targets(current)}"
            )

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if status is terminal (no further transitions)."""
        return len(cls._TRANSITIONS.get(status, set())) == 0

    @classmethod
    def all_statuses(cls) -> List[str]:
        """Return all valid case statuses."""
        return [s.value for s in CaseStatus]