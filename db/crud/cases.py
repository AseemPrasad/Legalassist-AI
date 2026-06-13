"""CRUD operations for Case management."""

from typing import List, Optional

import datetime as dt
from sqlalchemy.orm import Session

from db.models import (
    Case,
    CaseDocument,
    CaseDeadline,
    CaseTimeline,
    CaseStatus,
    CaseOutcome,
    Attachment,
    UserFeedback,
    ModelFeedback,
    SimilarityFeedback,
    CaseAnalytics,
)


def create_case(
    db: Session,
    user_id: int,
    case_type: str,
    title: str,
    description: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    case_number: Optional[str] = None,
) -> Case:
    """Create a new case for a user."""
    case = Case(
        user_id=user_id,
        case_type=case_type,
        title=title,
        description=description,
        jurisdiction=jurisdiction,
        case_number=case_number or f"CASE-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')}",
        status=CaseStatus.ACTIVE,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def get_user_cases(db: Session, user_id: int) -> List[Case]:
    """Get all cases for a user."""
    return db.query(Case).filter(Case.user_id == user_id).order_by(Case.created_at.desc()).all()


def get_case_by_id(db: Session, case_id: int) -> Optional[Case]:
    """Get a case by ID."""
    return db.query(Case).filter(Case.id == case_id).first()


def get_case_by_number(db: Session, case_number: str) -> Optional[Case]:
    """Get a case by case number."""
    return db.query(Case).filter(Case.case_number == case_number).first()


def update_case_status(db: Session, case_id: int, status: CaseStatus) -> Optional[Case]:
    """Update the status of a case."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if case:
        case.status = status
        db.commit()
        db.refresh(case)
    return case


def delete_case(db: Session, case_id: int) -> bool:
    """Delete a case and its associated data."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        return False

    # Delete associated data
    db.query(CaseDocument).filter(CaseDocument.case_id == case_id).delete(synchronize_session=False)
    db.query(CaseTimeline).filter(CaseTimeline.case_id == case_id).delete(synchronize_session=False)
    db.query(CaseDeadline).filter(CaseDeadline.case_id == case_id).delete(synchronize_session=False)
    db.query(Attachment).filter(Attachment.case_id == case_id).delete(synchronize_session=False)

    db.delete(case)
    db.commit()
    return True


def create_case_document(
    db: Session,
    case_id: int,
    document_type: str,
    document_content: Optional[str] = None,
    file_path: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
) -> CaseDocument:
    """Create a new document for a case."""
    doc = CaseDocument(
        case_id=case_id,
        document_type=document_type,
        document_content=document_content,
        file_path=file_path,
        summary=summary,
        remedies=remedies,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_case_documents(db: Session, case_id: int) -> List[CaseDocument]:
    """Get all documents for a case."""
    return db.query(CaseDocument).filter(
        CaseDocument.case_id == case_id
    ).order_by(CaseDocument.uploaded_at.desc()).all()


def get_case_timeline(db: Session, case_id: int) -> List[CaseTimeline]:
    """Get all timeline events for a case."""
    return db.query(CaseTimeline).filter(
        CaseTimeline.case_id == case_id
    ).order_by(CaseTimeline.event_date.desc()).all()


def create_timeline_event(
    db: Session,
    case_id: int,
    event_type: str,
    description: str,
    event_date: Optional[dt.datetime] = None,
    event_metadata: Optional[dict] = None,
) -> CaseTimeline:
    """Create a new timeline event."""
    event = CaseTimeline(
        case_id=case_id,
        event_type=event_type,
        description=description,
        event_date=event_date or dt.datetime.now(dt.timezone.utc),
        event_metadata=event_metadata,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def create_attachment(
    db: Session,
    user_id: int,
    case_id: int,
    original_filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
) -> Attachment:
    """Create a new attachment for a case."""
    attachment = Attachment(
        user_id=user_id,
        case_id=case_id,
        original_filename=original_filename,
        stored_path=stored_path,
        content_type=content_type,
        size_bytes=size_bytes,
    )
    db.add(attachment)
    db.commit()
    db.refresh(attachment)
    return attachment


def get_attachments_for_case(db: Session, case_id: int) -> List[Attachment]:
    """Get all attachments for a case."""
    return db.query(Attachment).filter(Attachment.case_id == case_id).order_by(Attachment.uploaded_at.desc()).all()


def get_user_stats(db: Session, user_id: int) -> dict:
    """Get statistics for a user's cases."""
    cases = db.query(Case).filter(Case.user_id == user_id).all()

    return {
        "total_cases": len(cases),
        "active_cases": len([c for c in cases if c.status == CaseStatus.ACTIVE]),
        "closed_cases": len([c for c in cases if c.status == CaseStatus.CLOSED]),
        "cases_by_type": {},
    }