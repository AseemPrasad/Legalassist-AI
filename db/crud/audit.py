from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from db.models import AuditEvent

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d[\d\s().-]{6,}\d))")
SENSITIVE_KEYS = {
    "password",
    "otp",
    "token",
    "secret",
    "document_content",
    "text",
    "summary",
    "message",
    "file_content",
    "content",
    "email",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, str):
        normalized = EMAIL_PATTERN.sub("[redacted-email]", value)
        normalized = PHONE_PATTERN.sub("[redacted-phone]", normalized)
        return normalized[:240]
    return str(value)[:240]


def _sanitize_metadata_value(value: Any, key: Optional[str] = None) -> Any:
    key_name = (key or "").lower()
    if key_name in SENSITIVE_KEYS:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): _sanitize_metadata_value(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_metadata_value(item, key_name) for item in value]
    return _safe_scalar(value)


def sanitize_audit_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        sanitized[str(key)] = _sanitize_metadata_value(value, str(key))
    return sanitized


def record_audit_event(
    db: Session,
    *,
    actor: str,
    action: str,
    resource: str,
    case_id: Optional[int] = None,
    actor_user_id: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    event = AuditEvent(
        actor=str(actor),
        actor_user_id=actor_user_id,
        action=str(action),
        resource=str(resource),
        case_id=case_id,
        occurred_at=_utcnow(),
        event_metadata=sanitize_audit_metadata(metadata),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def list_audit_events(
    db: Session,
    *,
    case_id: Optional[int] = None,
    actor_user_id: Optional[int] = None,
    limit: int = 100,
) -> List[AuditEvent]:
    query = db.query(AuditEvent)
    if case_id is not None:
        query = query.filter(AuditEvent.case_id == case_id)
    if actor_user_id is not None:
        query = query.filter(AuditEvent.actor_user_id == actor_user_id)
    return query.order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc()).limit(limit).all()


def audit_events_to_csv(events: Iterable[AuditEvent]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "actor", "actor_user_id", "action", "resource", "case_id", "occurred_at", "metadata"])
    for event in events:
        writer.writerow(
            [
                event.id,
                event.actor,
                event.actor_user_id,
                event.action,
                event.resource,
                event.case_id,
                event.occurred_at.isoformat() if event.occurred_at else None,
                json.dumps(event.event_metadata or {}, ensure_ascii=False, sort_keys=True),
            ]
        )
    return buffer.getvalue().encode("utf-8")
