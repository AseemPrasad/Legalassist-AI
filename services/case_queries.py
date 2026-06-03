"""Read-only case aggregation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import Attachment, Case, CaseDeadline, CaseDocument, CaseTimeline
from db.models.dtos import CaseSummaryDTO, CaseDetailDTO, DocumentDTO, DeadlineDTO, TimelineDTO, AttachmentDTO
from db.repositories.case_queries import fetch_case_summary_data_batch, fetch_case_detail_data_batch
from .timeline_service import timeline_service
from db.crud.audit import record_audit_event, record_immutable_audit_event


def _audit_case_view(func):
    @wraps(func)
    def wrapper(db: Session, user_id: int, case_id: int):
        result = func(db, user_id, case_id)
        if result is not None:
            payload = result if isinstance(result, dict) else {}
            record_immutable_audit_event(
                event_type="case.viewed",
                action="viewed",
                actor_user_id=user_id,
                resource_type="case",
                resource_id=str(case_id),
                outcome="success",
                case_id=case_id,
                metadata={
                    "documents": len(payload.get("documents", [])) if isinstance(payload.get("documents"), list) else None,
                    "deadlines": len(payload.get("deadlines", [])) if isinstance(payload.get("deadlines"), list) else None,
                    "timeline": len(payload.get("timeline", [])) if isinstance(payload.get("timeline"), list) else None,
                    "attachments": len(payload.get("attachments", [])) if isinstance(payload.get("attachments"), list) else None,
                },
            )
        return result

    return wrapper


def get_user_cases_summary(db: Session, user_id: int, include_closed: bool = True) -> List[Dict[str, Any]]:
    """
    Get summary of all cases for a user.
    
    Optimized to use batched queries instead of N+1 queries.
    Fetches latest document, next deadline, and document count per case in batch.
    """
    cases = db.query(Case).filter(Case.user_id == user_id).all()
    if not include_closed:
        cases = [case for case in cases if case.status.value != "closed"]

    if not cases:
        return []
    
    case_ids = [case.id for case in cases]
    
    # Fetch all related data in batch (3 queries instead of 3*N)
    latest_docs, next_deadlines, doc_counts = fetch_case_summary_data_batch(db, case_ids)
    
    # Build DTOs and convert to dicts
    summaries = []
    for case in cases:
        dto = CaseSummaryDTO.from_case_and_data(
            case,
            latest_docs.get(case.id),
            next_deadlines.get(case.id),
            doc_counts.get(case.id, 0),
        )
        summaries.append(dto.to_dict())

    return summaries


@_audit_case_view
def get_case_detail(db: Session, user_id: int, case_id: int) -> Optional[Dict[str, Any]]:
    """
    Get complete case details.
    
    Fetches case, documents, deadlines, timeline, and attachments.
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case or case.user_id != user_id:
        return None

    # Fetch all related data in batch
    documents, deadlines, timeline, attachments = fetch_case_detail_data_batch(db, case_id)
    
    # Build DTOs
    docs_list = [DocumentDTO.from_entity(doc) for doc in documents]
    deadlines_list = [DeadlineDTO.from_entity(d) for d in deadlines]
    timeline_list = [TimelineDTO.from_entity(t) for t in timeline]
    attachments_list = [AttachmentDTO.from_entity(a) for a in attachments]
    
    # Get latest document's remedies
    remedies = documents[0].remedies if documents else None
    
    # Get timeline events via service (keeps existing behavior)
    timeline_events = timeline_service.get_case_timeline_events(db, case_id)
    
    # Build and return detail DTO
    detail = CaseDetailDTO(
        case={
            "id": case.id,
            "case_number": case.case_number,
            "title": case.title,
            "case_type": case.case_type,
            "jurisdiction": case.jurisdiction,
            "status": case.status.value,
            "created_at": case.created_at.isoformat(),
        },
        documents=docs_list,
        deadlines=deadlines_list,
        attachments=attachments_list,
        timeline=timeline_events,
        remedies=remedies,
    )

    record_audit_event(
        db,
        actor=f"user:{user_id}",
        actor_user_id=user_id,
        action="view_case_detail",
        resource=f"case:{case_id}",
        case_id=case_id,
        metadata={
            "documents": len(docs_list),
            "deadlines": len(deadlines_list),
            "timeline_events": len(timeline_events),
            "attachments": len(attachments_list),
        },
    )
    
    return detail.to_dict()


def generate_case_summary_text(db: Session, user_id: int, case_id: int) -> Optional[str]:
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case or case.user_id != user_id:
        return None

    documents = db.query(CaseDocument).filter(CaseDocument.case_id == case_id).all()
    timeline = db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).all()
    deadlines = db.query(CaseDeadline).filter(CaseDeadline.case_id == case_id).order_by(CaseDeadline.deadline_date).all()

    lines = [
        "=" * 60,
        f"CASE SUMMARY: {case.case_number}",
        "=" * 60,
        "",
        f"Title: {case.title or 'N/A'}",
        f"Type: {case.case_type}",
        f"Jurisdiction: {case.jurisdiction}",
        f"Status: {case.status.value}",
        f"Created: {case.created_at.strftime('%d %B %Y')}",
        "",
        "-" * 60,
        "DOCUMENTS",
        "-" * 60,
    ]

    for doc in documents:
        lines.append(f"\n[{doc.document_type.value}] - {doc.uploaded_at.strftime('%d %B %Y')}")
        if doc.summary:
            lines.append(f"Summary: {doc.summary}")

    lines.extend(["", "-" * 60, "TIMELINE", "-" * 60])
    for event in timeline:
        lines.append(f"[{event.event_date.strftime('%d %B %Y')}] {event.event_type}: {event.description}")

    lines.extend(["", "-" * 60, "DEADLINES", "-" * 60])
    for d in deadlines:
        status = "✓" if d.is_completed else "○"
        lines.append(f"[{status}] {d.deadline_type}: {d.deadline_date.strftime('%d %B %Y')} - {d.description or 'No description'}")

    lines.extend([
        "",
        "=" * 60,
        f"Generated: {datetime.now(timezone.utc).strftime('%d %B %Y %H:%M')}",
        "=" * 60,
    ])

    return "\n".join(lines)
