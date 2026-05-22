"""Pydantic schemas for timeline event payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from core.time_serialization import to_utc_iso


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return to_utc_iso(value)
    if isinstance(value, dict):
        return {key: _json_safe_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [_json_safe_value(item) for item in value]
    return value


class TimelineEventPayload(BaseModel):
    """Validated realtime payload for a single timeline event."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["timeline_event"]
    case_id: int
    event_type: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_id: int

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_serializer("metadata")
    def serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return _json_safe_value(metadata)