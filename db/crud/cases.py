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


def get_case_document_by_id(db: Session, document_id: int) -> Optional[CaseDocument]:
    """Get a case document by ID."""
    return db.query(CaseDocument).filter(CaseDocument.id == document_id).first()


def create_case_record(db: Session, case_id: int, record_type: str, record_data: dict) -> "CaseRecord":
    """Create a case record."""
    from db.models import CaseRecord
    
    record = CaseRecord(case_id=case_id, record_type=record_type, record_data=record_data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_case_record(db: Session, record_id: int) -> Optional["CaseRecord"]:
    """Get a case record by ID."""
    from db.models import CaseRecord
    return db.query(CaseRecord).filter(CaseRecord.id == record_id).first()


def get_cases_by_criteria(
    db: Session,
    user_id: Optional[int] = None,
    status: Optional[CaseStatus] = None,
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
) -> List[Case]:
    """Get cases matching criteria."""
    query = db.query(Case)
    
    if user_id is not None:
        query = query.filter(Case.user_id == user_id)
    if status is not None:
        query = query.filter(Case.status == status)
    if case_type is not None:
        query = query.filter(Case.case_type == case_type)
    if jurisdiction is not None:
        query = query.filter(Case.jurisdiction == jurisdiction)
    
    return query.order_by(Case.created_at.desc()).all()


def update_case_outcome(db: Session, case_id: int, outcome: CaseOutcome, notes: Optional[str] = None) -> Optional[Case]:
    """Update the outcome of a case."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if case:
        case.outcome = outcome
        if notes:
            case.description = (case.description or "") + f"\n\nOutcome notes: {notes}"
        db.commit()
        db.refresh(case)
    return case


def submit_user_feedback(
    db: Session,
    case_id: int,
    user_id: int,
    rating: int,
    feedback_text: Optional[str] = None,
) -> UserFeedback:
    """Submit user feedback for a case."""
    feedback = UserFeedback(
        case_id=case_id,
        user_id=user_id,
        rating=rating,
        feedback_text=feedback_text,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_user_feedback(db: Session, case_id: int) -> List[UserFeedback]:
    """Get all feedback for a case."""
    return db.query(UserFeedback).filter(UserFeedback.case_id == case_id).order_by(UserFeedback.created_at.desc()).all()


def submit_model_feedback(
    db: Session,
    case_id: int,
    user_id: int,
    model_id: str,
    feedback_type: str,
    score: float,
) -> ModelFeedback:
    """Submit model performance feedback."""
    feedback = ModelFeedback(
        case_id=case_id,
        user_id=user_id,
        model_id=model_id,
        feedback_type=feedback_type,
        score=score,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


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


def get_similarity_feedback(
    db: Session,
    user_id: Optional[str] = None,
    query_signature: Optional[str] = None,
    candidate_case_id: Optional[int] = None,
    limit: int = 100,
) -> List[SimilarityFeedback]:
    """Get similarity feedback rows filtered by criteria."""
    query = db.query(SimilarityFeedback)

    if user_id is not None:
        query = query.filter(SimilarityFeedback.user_id == str(user_id))
    if query_signature is not None:
        query = query.filter(SimilarityFeedback.query_signature == query_signature)
    if candidate_case_id is not None:
        query = query.filter(SimilarityFeedback.candidate_case_id == candidate_case_id)

    return query.order_by(SimilarityFeedback.created_at.desc()).limit(limit).all()