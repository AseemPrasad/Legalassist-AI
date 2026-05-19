"""Natural-language deadline parsing and automatic deadline creation."""

from __future__ import annotations

import datetime as dt
import re
from typing import Dict, Optional

from sqlalchemy.orm import Session

from db.models import CaseDeadline, CaseTimeline
from .timeline_service import timeline_service


_APPEAL_CONTEXT = r"(?:file(?:\s+an?)?\s+appeal|appeal|notice(?:\s+of)?\s+appeal|challenge(?:\s+an?)?(?:\s+order)?)"
_APPEAL_DAY_PATTERNS = (
    re.compile(
        rf"\b(?P<context>{_APPEAL_CONTEXT})\b(?:\W+\w+){{0,8}}?\s*(?:about\s+|approximately\s+)?(?P<days>\d{{1,3}})\s*[,.-]?\s*(?:(?:business|calendar)\s+)?day(?:s)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<days>\d{{1,3}})\s*[,.-]?\s*(?:(?:business|calendar)\s+)?day(?:s)?\b\s*(?:to\s+)?(?P<context>{_APPEAL_CONTEXT})\b",
        re.IGNORECASE,
    ),
)


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
    remedies: Dict,
    document_id: int,
) -> None:
    appeal_days = remedies.get("appeal_days")
    if not appeal_days:
        return

    appeal_days_str = str(appeal_days).strip()
    days = _extract_days_from_text(appeal_days_str)
    if days is None or not _validate_days_value(days):
        return

    if _has_matching_deadline_creation(db, case_id, days, document_id):
        return

    current_time = dt.datetime.now(dt.timezone.utc)
    deadline_date = current_time + dt.timedelta(days=days)

    deadline = CaseDeadline(
        user_id=user_id,
        case_id=case_id,
        case_title=case_title,
        deadline_date=deadline_date,
        deadline_type="appeal",
        description=f"Appeal deadline - {remedies.get('appeal_court', 'Unknown court')}",
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
