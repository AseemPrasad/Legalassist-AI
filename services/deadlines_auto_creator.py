"""Natural-language deadline parsing and automatic deadline creation."""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from db.models import CaseDeadline, CaseTimeline
from .timeline_service import timeline_service


logger = logging.getLogger(__name__)


_APPEAL_CONTEXT = r"(?:file(?:\s+an?)?\s+appeal|appeal|notice(?:\s+of)?\s+appeal|challenge(?:\s+an?)?(?:\s+order)?)"
_APPEAL_SEPARATOR = r"(?:[\s:;,\-().\[\]{}]+(?:(?!\d{1,3}\b)\w+\s+))*[\s:;,\-().\[\]{}]*"
_APPEAL_DAY_PATTERNS = (
    re.compile(
        rf"\b(?P<context>{_APPEAL_CONTEXT})\b{_APPEAL_SEPARATOR}(?:about\s+|approximately\s+)?(?P<days>\d{{1,3}})\s*[,.-]?\s*(?:(?:business|calendar)\s+)?day(?:s)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<days>\d{{1,3}})\s*[,.-]?\s*(?:(?:business|calendar)\s+)?day(?:s)?\b\s*(?:to\s+)?(?P<context>{_APPEAL_CONTEXT})\b",
        re.IGNORECASE,
    ),
)


class RemediesPayload(BaseModel):
    appeal_days: Optional[Union[int, str]] = None
    appeal_court: Optional[str] = None


def _validate_remedies_payload(remedies: Any) -> Optional[RemediesPayload]:
    if not remedies:
        return None

    if not isinstance(remedies, dict):
        logger.warning(
            "Skipping deadline creation: remedies payload must be a mapping, got %s",
            type(remedies).__name__,
        )
        return None

    try:
        payload = RemediesPayload.model_validate(remedies)
    except ValidationError as exc:
        logger.warning("Skipping deadline creation: invalid remedies payload shape: %s", exc)
        return None

    if payload.appeal_days is None:
        logger.warning("Skipping deadline creation: remedies payload exists but appeal_days is missing")
        return None

    if isinstance(payload.appeal_days, str) and not payload.appeal_days.strip():
        logger.warning("Skipping deadline creation: remedies payload exists but appeal_days is empty")
        return None

    return payload


def _emit_deadline_skip_event(
    db: Session,
    case_id: int,
    reason: str,
    metadata: Dict[str, Any],
) -> None:
    try:
        event_metadata = dict(metadata)
        event_metadata["reason"] = reason
        timeline_service.create_event(
            db=db,
            case_id=case_id,
            event_type="deadline_skipped",
            description=f"Deadline creation skipped: {reason}",
            metadata=event_metadata,
        )
    except Exception:
        logger.exception("Failed to record deadline_skipped timeline event")


def _log_deadline_skip(
    db: Session,
    case_id: int,
    reason: str,
    **metadata: Any,
) -> None:
    safe_metadata = {k: v for k, v in metadata.items() if v is not None}
    logger.warning(
        "Skipping deadline creation: %s | metadata=%s",
        reason,
        safe_metadata,
    )
    _emit_deadline_skip_event(db, case_id, reason, safe_metadata)


def _extract_days_from_text(text: str) -> Optional[int]:
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    if text.isdigit():
        return int(text)

    for pattern in _APPEAL_DAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group("days"))

    return None


def _validate_days_value(days: int) -> bool:
    return 1 <= days <= 365


def _has_matching_deadline_creation(db: Session, case_id: int, days: int, document_id: int) -> bool:
    events = db.query(CaseTimeline).filter(
        CaseTimeline.case_id == case_id,
        CaseTimeline.event_type == "deadline_created",
    ).all()

    for event in events:
        metadata = event.event_metadata if isinstance(event.event_metadata, dict) else {}
        if metadata.get("source_days") == days:
            return True
        if document_id is not None and metadata.get("document_id") == document_id:
            return True

    return False


def auto_create_deadlines_from_remedies(
    db: Session,
    user_id: int,
    case_id: int,
    case_title: str,
    remedies: Any,
    document_id: int,
) -> None:
    validated_remedies = _validate_remedies_payload(remedies)
    if validated_remedies is None:
        return

    appeal_days = validated_remedies.appeal_days
    if appeal_days is None:
        _log_deadline_skip(
            db,
            case_id,
            "appeal_days_missing",
            user_id=user_id,
            case_title=case_title,
            document_id=document_id,
            remedies_present=True,
        )
        return

    appeal_days_str = str(appeal_days).strip()
    days = _extract_days_from_text(appeal_days_str)
    if days is None or not _validate_days_value(days):
        _log_deadline_skip(
            db,
            case_id,
            "appeal_days_invalid",
            user_id=user_id,
            case_title=case_title,
            document_id=document_id,
            appeal_days_type=type(appeal_days).__name__,
            appeal_days_value=appeal_days_str[:120],
        )
        return

    if _has_matching_deadline_creation(db, case_id, days, document_id):
        _log_deadline_skip(
            db,
            case_id,
            "matching_deadline_exists",
            user_id=user_id,
            case_title=case_title,
            document_id=document_id,
            source_days=days,
        )
        return

    current_time = dt.datetime.now(dt.timezone.utc)
    deadline_date = current_time + dt.timedelta(days=days)

    deadline = CaseDeadline(
        user_id=user_id,
        case_id=case_id,
        case_title=case_title,
        deadline_date=deadline_date,
        deadline_type="appeal",
        description=f"Appeal deadline - {validated_remedies.appeal_court or 'Unknown court'}",
    )
    db.add(deadline)
    db.flush()

    timeline_service.create_event(
        db=db,
        case_id=case_id,
        event_type="deadline_created",
        description=f"Appeal deadline set for {deadline_date.strftime('%d %B %Y')} based on document analysis",
        metadata={
            "deadline_id": deadline.id,
            "document_id": document_id,
            "source_days": days,
            "original_text": appeal_days_str,
        },
    )
