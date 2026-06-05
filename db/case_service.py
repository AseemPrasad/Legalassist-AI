from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import (
    Case,
    CaseDeadline,
    CaseDocument,
    CaseOutcome,
    CaseRecord,
    CaseStatus,
    CaseTimeline,
    CaseNote,
    CaseNoteVersion,
    ModelFeedback,
    UserFeedback,
)
from db.crud.knowledge import record_knowledge_invalidation


def create_case(db: Session, user_id: int, case_number: str, case_type: str, jurisdiction: str, title: Optional[str] = None) -> Case:
    """Create a new case"""
    case = Case(
        user_id=user_id,
        case_number=case_number,
        case_type=case_type,
        jurisdiction=jurisdiction,
        title=title,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def get_user_cases(db: Session, user_id: int) -> List[Case]:
    """Get all cases for a user"""
    return db.query(Case).filter(Case.user_id == user_id).order_by(Case.created_at.desc()).all()


def get_case_by_id(db: Session, case_id: int) -> Optional[Case]:
    """Get a case by ID"""
    return db.query(Case).filter(Case.id == case_id).first()


def get_case_by_number(db: Session, user_id: int, case_number: str) -> Optional[Case]:
    """Get a case by case number for a specific user"""
    return db.query(Case).filter(
        Case.user_id == user_id,
        Case.case_number == case_number,
    ).first()


def update_case_status(db: Session, case_id: int, status: CaseStatus) -> Optional[Case]:
    """Update case status"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if case:
        case.status = status
        db.commit()
        db.refresh(case)
    return case


def delete_case(db: Session, case_id: int) -> bool:
    """Delete a case and all related data"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if case:
        db.delete(case)
        db.commit()
        return True
    return False


def create_case_document(
    db: Session,
    case_id: int,
    document_type,
    user_id: int,
    document_content: Optional[str] = None,
    file_path: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
    extracted_metadata: Optional[dict] = None,
    extraction_method: Optional[str] = None,
    ocr_used: bool = False,
    source_attachment_id: Optional[int] = None,
) -> CaseDocument:
    """Create a new case document"""
    doc = CaseDocument(
        case_id=case_id,
        source_attachment_id=source_attachment_id,
        document_type=document_type,
        document_content=document_content,
        file_path=file_path,
        summary=summary,
        remedies=remedies,
        extracted_metadata=extracted_metadata,
        extraction_method=extraction_method,
        ocr_used=ocr_used,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    record_knowledge_invalidation(
        db,
        scope_type="case",
        case_id=case_id,
        document_id=doc.id,
        user_id=user_id,
        reason="document_created",
        details={
            "document_type": getattr(document_type, "value", document_type),
            "source_attachment_id": source_attachment_id,
            "changed_fields": ["document_content", "summary", "remedies", "extracted_metadata"],
        },
    )
    return doc


def get_case_documents(db: Session, case_id: int) -> List[CaseDocument]:
    """Get all documents for a case"""
    return db.query(CaseDocument).filter(
        CaseDocument.case_id == case_id
    ).order_by(CaseDocument.uploaded_at).all()


def get_case_document_by_id(db: Session, document_id: int) -> Optional[CaseDocument]:
    """Get a document by ID"""
    return db.query(CaseDocument).filter(CaseDocument.id == document_id).first()


def get_case_note(db: Session, case_id: int, user_id: int) -> Optional[CaseNote]:
    """Get the editable note state for a case owned by the user."""
    return (
        db.query(CaseNote)
        .filter(
            CaseNote.case_id == case_id,
            CaseNote.user_id == user_id,
        )
        .first()
    )


def save_case_note_draft(
    db: Session,
    case_id: int,
    user_id: int,
    note_text: str,
    changed_by_email: Optional[str] = None,
) -> CaseNote:
    """Persist the current draft text without creating a published version."""
    case = get_case_by_id(db, case_id)
    if not case or case.user_id != user_id:
        raise ValueError("Case not found or not owned by user")

    note = get_case_note(db, case_id, user_id)
    if not note:
        note = CaseNote(case_id=case_id, user_id=user_id, draft_text=note_text)
        db.add(note)
    else:
        note.draft_text = note_text
        note.draft_updated_at = dt.datetime.now(dt.timezone.utc)

    db.commit()
    db.refresh(note)
    return note


def publish_case_note(
    db: Session,
    case_id: int,
    user_id: int,
    note_text: Optional[str] = None,
    changed_by_email: Optional[str] = None,
) -> CaseNoteVersion:
    """Create an immutable published version from the current draft."""
    case = get_case_by_id(db, case_id)
    if not case or case.user_id != user_id:
        raise ValueError("Case not found or not owned by user")

    note = get_case_note(db, case_id, user_id)
    if not note:
        note = CaseNote(case_id=case_id, user_id=user_id, draft_text=note_text or "")
        db.add(note)
        db.flush()
    elif note_text is not None:
        note.draft_text = note_text
        note.draft_updated_at = dt.datetime.now(dt.timezone.utc)

    current_text = note_text if note_text is not None else note.draft_text
    if current_text is None:
        current_text = ""

    next_version = (
        db.query(CaseNoteVersion.version_number)
        .filter(CaseNoteVersion.case_id == case_id, CaseNoteVersion.note_id == note.id)
        .order_by(CaseNoteVersion.version_number.desc())
        .first()
    )
    version_number = (next_version[0] if next_version else 0) + 1

    version = CaseNoteVersion(
        note_id=note.id,
        case_id=case_id,
        version_number=version_number,
        note_text=current_text,
        change_type="published",
        changed_by_user_id=user_id,
        changed_by_email=changed_by_email,
        version_metadata={"published_from_draft": True},
    )
    note.published_text = current_text
    note.published_at = dt.datetime.now(dt.timezone.utc)
    note.published_version_id = version_number

    db.add(version)
    db.commit()
    db.refresh(version)
    db.refresh(note)
    return version


def get_case_note_history(db: Session, case_id: int, user_id: int) -> List[CaseNoteVersion]:
    """Get immutable published note versions for a case."""
    case = get_case_by_id(db, case_id)
    if not case or case.user_id != user_id:
        return []

    return (
        db.query(CaseNoteVersion)
        .filter(CaseNoteVersion.case_id == case_id)
        .order_by(CaseNoteVersion.version_number.desc(), CaseNoteVersion.created_at.desc())
        .all()
    )


def update_case_document(
    db: Session,
    document_id: int,
    document_content: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
    extracted_metadata: Optional[dict] = None,
    extraction_method: Optional[str] = None,
    ocr_used: Optional[bool] = None,
) -> Optional[CaseDocument]:
    """Update case document"""
    doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
    if doc:
        changed_fields = []
        if document_content is not None:
            doc.document_content = document_content
            changed_fields.append("document_content")
        if summary is not None:
            doc.summary = summary
            changed_fields.append("summary")
        if remedies is not None:
            doc.remedies = remedies
            changed_fields.append("remedies")
        if extracted_metadata is not None:
            doc.extracted_metadata = extracted_metadata
            changed_fields.append("extracted_metadata")
        if extraction_method is not None:
            doc.extraction_method = extraction_method
            changed_fields.append("extraction_method")
        if ocr_used is not None:
            doc.ocr_used = ocr_used
            changed_fields.append("ocr_used")
        try:
            db.commit()
            db.refresh(doc)
            if changed_fields:
                reason = f"{changed_fields[0]}_updated"
                record_knowledge_invalidation(
                    db,
                    scope_type="case",
                    case_id=doc.case_id,
                    document_id=doc.id,
                    reason=reason,
                    details={
                        "changed_fields": changed_fields,
                        "document_id": doc.id,
                        "case_id": doc.case_id,
                    },
                )
        except Exception as e:
            db.rollback()
            raise RuntimeError(f"Database write failed for case document {document_id}: {str(e)}") from e
    return doc


def create_case_record(
    db: Session,
    hashed_case_id: str,
    case_type: str,
    jurisdiction: str,
    court_name: Optional[str] = None,
    judge_name: Optional[str] = None,
    plaintiff_type: Optional[str] = None,
    defendant_type: Optional[str] = None,
    case_value: Optional[str] = None,
    outcome: Optional[str] = None,
    judgment_summary: Optional[str] = None,
) -> CaseRecord:
    """Create a new case record for analytics"""
    case = CaseRecord(
        hashed_case_id=hashed_case_id,
        case_type=case_type,
        jurisdiction=jurisdiction,
        court_name=court_name,
        judge_name=judge_name,
        plaintiff_type=plaintiff_type,
        defendant_type=defendant_type,
        case_value=case_value,
        outcome=outcome,
        judgment_summary=judgment_summary,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def get_case_record(db: Session, hashed_case_id: str) -> Optional[CaseRecord]:
    """Get a case record by hashed ID"""
    return db.query(CaseRecord).filter(CaseRecord.hashed_case_id == hashed_case_id).first()


ALLOWED_CASE_FILTER_FIELDS = frozenset({
    "case_type",
    "jurisdiction",
    "court_name",
    "judge_name",
    "plaintiff_type",
    "defendant_type",
    "outcome",
})


def get_cases_by_criteria(db: Session, **criteria) -> List[CaseRecord]:
    """Search case records by approved criteria fields only."""
    query = db.query(CaseRecord)
    for key, value in criteria.items():
        if key not in ALLOWED_CASE_FILTER_FIELDS:
            continue
        if hasattr(CaseRecord, key) and value:
            query = query.filter(getattr(CaseRecord, key) == value)
    return query.all()


def update_case_outcome(
    db: Session,
    hashed_case_id: str,
    appeal_filed: bool = False,
    appeal_date: Optional[dt.datetime] = None,
    appeal_outcome: Optional[str] = None,
    appeal_success: Optional[bool] = None,
    time_to_appeal_verdict: Optional[int] = None,
    appeal_cost: Optional[str] = None,
    additional_notes: Optional[str] = None,
) -> CaseOutcome:
    """Update or create case outcome data"""
    record = get_case_record(db, hashed_case_id)
    if not record:
        raise ValueError(f"Case {hashed_case_id} not found")

    outcome = db.query(CaseOutcome).filter(CaseOutcome.case_id == record.id).first()
    if not outcome:
        outcome = CaseOutcome(case_id=record.id)
        db.add(outcome)

    outcome.appeal_filed = appeal_filed
    if appeal_date:
        outcome.appeal_date = appeal_date
    if appeal_outcome:
        outcome.appeal_outcome = appeal_outcome
    if appeal_success is not None:
        outcome.appeal_success = appeal_success
    if time_to_appeal_verdict:
        outcome.time_to_appeal_verdict = time_to_appeal_verdict
    if appeal_cost:
        outcome.appeal_cost = appeal_cost
    if additional_notes:
        outcome.additional_notes = additional_notes

    db.commit()
    db.refresh(outcome)
    return outcome


def get_user_stats(db: Session, user_id: int) -> dict:
    """Calculate high-level stats for a user dashboard"""
    cases = get_user_cases(db, user_id)

    active_count = len([c for c in cases if c.status == CaseStatus.ACTIVE])
    appealed_count = len([c for c in cases if c.status == CaseStatus.APPEALED])
    closed_count = len([c for c in cases if c.status == CaseStatus.CLOSED])

    now = dt.datetime.now(dt.timezone.utc)
    upcoming_deadlines = db.query(CaseDeadline).filter(
        CaseDeadline.user_id == user_id,
        CaseDeadline.is_completed.is_(False),
        CaseDeadline.deadline_date > now,
    ).count()

    return {
        "total_cases": len(cases),
        "active_cases": active_count,
        "appealed_cases": appealed_count,
        "closed_cases": closed_count,
        "upcoming_deadlines": upcoming_deadlines,
    }


def submit_model_feedback(
    db: Session,
    user_id: str,
    model_name: str,
    task: str,
    case_id: Optional[int] = None,
    is_accurate: Optional[bool] = None,
    corrected_text: Optional[str] = None,
    feedback_notes: Optional[str] = None,
) -> ModelFeedback:
    """Submit model output feedback"""
    fb = ModelFeedback(
        user_id=str(user_id),
        model_name=model_name,
        task=task,
        case_id=case_id,
        is_accurate=is_accurate,
        corrected_text=corrected_text,
        feedback_notes=feedback_notes,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb


def get_case_timeline(db: Session, case_id: int) -> List[CaseTimeline]:
    """Get all timeline events for a case"""
    return db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).order_by(CaseTimeline.event_date.desc()).all()


def create_timeline_event(
    db: Session,
    case_id: int,
    event_type: str,
    description: str,
    event_date: Optional[dt.datetime] = None,
    metadata: Optional[dict] = None,
) -> CaseTimeline:
    """Create a new timeline event"""
    event = CaseTimeline(
        case_id=case_id,
        event_type=event_type,
        description=description,
        event_date=event_date or dt.datetime.now(dt.timezone.utc),
        event_metadata=metadata,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def transition_deadline(db: Session, deadline_id: int, target_status: str, actor_user_id: int) -> CaseDeadline:
    """
    Transition a deadline to a new status with validation and audit trail.
    Allowed transitions:
    - active -> completed
    - active -> overdue
    - overdue -> completed
    - overdue -> active
    - completed -> active (reopening)
    """
    from db.crud.audit import record_audit_event

    deadline = db.query(CaseDeadline).filter(CaseDeadline.id == deadline_id).first()
    if not deadline:
        raise ValueError(f"Deadline {deadline_id} not found")

    old_status = deadline.status or ("completed" if deadline.is_completed else "active")
    if old_status == target_status:
        raise ValueError(f"Deadline is already in '{target_status}' status")

    # Validate transition
    allowed = {
        "active": {"completed", "overdue"},
        "overdue": {"completed", "active"},
        "completed": {"active"},
    }

    if target_status not in allowed.get(old_status, set()):
        raise ValueError(f"Invalid transition from '{old_status}' to '{target_status}'")

    deadline.status = target_status
    if target_status == "completed":
        deadline.is_completed = True
    else:
        deadline.is_completed = False

    db.commit()
    db.refresh(deadline)

    # Audit trail
    record_audit_event(
        db,
        actor=f"user:{actor_user_id}",
        action="transition_deadline",
        resource=f"deadline:{deadline.id}",
        case_id=deadline.case_id,
        actor_user_id=actor_user_id,
        metadata={
            "deadline_id": deadline.id,
            "old_status": old_status,
            "new_status": target_status,
        }
    )

    return deadline

