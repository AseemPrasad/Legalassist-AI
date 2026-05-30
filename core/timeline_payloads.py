"""Pydantic schemas for timeline event payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

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

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 2
    LEGACY_SCHEMA_VERSION: ClassVar[int] = 1

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(default=LEGACY_SCHEMA_VERSION, ge=1)
    type: Literal["timeline_event"] = "timeline_event"
    case_id: int
    event_type: str
    description: str = ""
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    event_id: int

    @model_validator(mode="before")
    @classmethod
    def normalize_wire_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if "schema_version" not in normalized:
            normalized["schema_version"] = normalized.pop("schemaVersion", normalized.pop("version", cls.LEGACY_SCHEMA_VERSION))

        alias_map = {
            "caseId": "case_id",
            "eventType": "event_type",
            "eventId": "event_id",
            "eventDate": "timestamp",
            "event_date": "timestamp",
            "messagePreview": "message_preview",
        }
        for old_key, new_key in alias_map.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized.pop(old_key)

        if normalized.get("metadata") is None:
            normalized["metadata"] = {}

        return normalized

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_serializer("metadata")
    def serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return _json_safe_value(metadata)


class TimelineSubscribedPayload(BaseModel):
    """Validated realtime payload for the initial websocket subscription message."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(default=TimelineEventPayload.CURRENT_SCHEMA_VERSION, ge=1)
    type: Literal["subscribed"] = "subscribed"
    case_id: int
