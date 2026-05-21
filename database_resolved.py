"""Compatibility shim for the original monolithic `database.py`.

The project has moved models and CRUD helpers into the `db/` package, but many
existing imports still point at `database`. This module re-exports the pieces
needed by the current codebase and keeps the authentication/OTP security path
working while the refactor continues.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, List
from sqlalchemy.orm import Session

from db.base import Base
from db.session import engine, SessionLocal, init_db, db_session, get_db, _to_utc_datetime, _datetime_for_db
from db.models import (
    User,
    OTPVerification,
    NotificationStatus,
    NotificationChannel,
    NotificationLog,
    NotificationTemplate,
    UserPreference,
    CaseDeadline,
    Case,
    CaseDocument,
    Attachment,
    CaseTimeline,
    CaseStatus,
    DocumentType,
    UserFeedback,
    CaseRecord,
    CaseOutcome,
    CaseAnalytics,
    ModelFeedback,
    ModelPerformance,
    ModelRoutingRule,
    SimilarityFeedback,
    RevokedToken,
    CaseEmbedding,
    CaseIssue,
    CaseArgument,
    KnowledgeGraphEdge,
    PrecedentMatch,
)
from db.crud.notifications import (
    create_case_deadline,
    get_upcoming_deadlines,
    has_notification_been_sent,
    log_notification,
    get_notification_history,
)

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "init_db",
    "db_session",
    "get_db",
    "_to_utc_datetime",
    "_datetime_for_db",
    "NotificationStatus",
    "NotificationChannel",
    "UserPreference",
    "NotificationLog",
    "NotificationTemplate",
    "CaseDeadline",
    "Case",
    "CaseDocument",
    "Attachment",
    "CaseTimeline",
    "CaseStatus",
    "DocumentType",
    "User",
    "OTPVerification",
    "UserFeedback",
    "CaseRecord",
    "CaseOutcome",
    "CaseAnalytics",
    "ModelFeedback",
    "ModelPerformance",
    "ModelRoutingRule",
    "SimilarityFeedback",
    "RevokedToken",
    "CaseEmbedding",
    "CaseIssue",
    "CaseArgument",
    "KnowledgeGraphEdge",
    "PrecedentMatch",
    "create_case_deadline",
    "get_upcoming_deadlines",
    "get_user_deadlines",
    "has_notification_been_sent",
    "log_notification",
    "get_notification_history",
    "create_or_update_user_preference",
    "create_user",
    "get_user_by_email",
    "update_user_last_login",
    "create_otp_verification",
    "get_pending_otp",
    "mark_otp_as_used",
    "is_email_locked_out",
    "record_otp_failed_attempt",
    "reset_otp_failed_attempts",
    "cleanup_expired_otps",
    "create_case",
    "get_user_cases",
    "get_case_by_id",
    "get_case_by_number",
    "update_case_status",
    "delete_case",
    "create_case_document",
    "get_case_documents",
    "get_case_document_by_id",
    "create_case_record",
    "get_case_record",
    "get_cases_by_criteria",
    "update_case_outcome",
    "submit_user_feedback",
    "get_user_feedback",
    "submit_model_feedback",
    "get_case_timeline",
    "create_timeline_event",
    "create_attachment",
    "get_attachments_for_case",
]


# ==================== Legacy Helper Functions ====================


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email address"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str) -> User:
    """Create a new user"""
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_last_login(db: Session, user_id: int) -> Optional[User]:
    """Update last login timestamp for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(user)
    return user


def create_otp_verification(db: Session, email: str, otp_hash: str, expires_at: dt.datetime) -> OTPVerification:
    """Create a new OTP verification record"""
    otp = OTPVerification(email=email, otp_hash=otp_hash, expires_at=expires_at)
    db.add(otp)
    db.commit()
    db.refresh(otp)
    return otp


def get_pending_otp(db: Session, email: str) -> Optional[OTPVerification]:
    """Get the latest unused, non-expired OTP for an email"""
    now = dt.datetime.now(dt.timezone.utc)
    return db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.is_used == False,
        OTPVerification.expires_at > now
    ).order_by(OTPVerification.created_at.desc()).first()


def mark_otp_as_used(db: Session, otp_id: int) -> bool:
    try:
        otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
        if otp:
            otp.is_used = True
            db.commit()
            db.refresh(otp)
            return True
        return False
    except Exception:
        db.rollback()
        return False


def is_email_locked_out(db: Session, email: str) -> Optional[dt.datetime]:
    """Check if email is currently locked out. Returns locked_until if locked, None otherwise."""
    lockout = db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.locked_until != None,
        OTPVerification.locked_until > dt.datetime.now(dt.timezone.utc)
    ).order_by(OTPVerification.locked_until.desc()).first()
    return lockout.locked_until if lockout else None


def record_otp_failed_attempt(db: Session, otp_id: int, lockout_duration_minutes: int = 15, max_failed_attempts: int = 5) -> bool:
    """Record failed OTP attempt with email-level lockout protection."""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.failed_attempts += 1
        if otp.failed_attempts >= max_failed_attempts:
            # Lock out at email level, not just OTP level
            lockout_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=lockout_duration_minutes)
            otp.locked_until = lockout_until
            
            # Also lock all other pending OTPs for same email
            db.query(OTPVerification).filter(
                OTPVerification.email == otp.email,
                OTPVerification.id != otp_id
            ).update({"locked_until": lockout_until})
        
        db.commit()
        db.refresh(otp)
        return True
    return False


def reset_otp_failed_attempts(db: Session, otp_id: int) -> bool:
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.failed_attempts = 0
        otp.locked_until = None
        db.commit()
        db.refresh(otp)
        return True
    return False


def cleanup_expired_otps(db: Session) -> int:
    """Delete expired OTP records"""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(OTPVerification).filter(OTPVerification.expires_at < now).delete()
    db.commit()
    return deleted


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
    document_type: DocumentType,
    user_id: int,
    document_content: Optional[str] = None,
    file_path: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
) -> CaseDocument:
    """Create a new case document"""
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


def get_cases_by_criteria(db: Session, **criteria) -> List[CaseRecord]:
    """Search case records by criteria"""
    query = db.query(CaseRecord)
    for key, value in criteria.items():
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
    if appeal_date: outcome.appeal_date = appeal_date
    if appeal_outcome: outcome.appeal_outcome = appeal_outcome
    if appeal_success is not None: outcome.appeal_success = appeal_success
    if time_to_appeal_verdict: outcome.time_to_appeal_verdict = time_to_appeal_verdict
    if appeal_cost: outcome.appeal_cost = appeal_cost
    if additional_notes: outcome.additional_notes = additional_notes

    db.commit()
    db.refresh(outcome)
    return outcome


def submit_user_feedback(
    db: Session,
    user_id: int,
    case_id: Optional[int] = None,
    did_appeal: Optional[bool] = None,
    appeal_outcome: Optional[str] = None,
    satisfaction_rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
) -> UserFeedback:
    """Submit user feedback"""
    feedback = UserFeedback(
        user_id=user_id,
        case_id=case_id,
        did_appeal=did_appeal,
        appeal_outcome=appeal_outcome,
        satisfaction_rating=satisfaction_rating,
        feedback_text=feedback_text,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_user_feedback(db: Session, user_id: int) -> List[UserFeedback]:
    """Get feedback history for a user"""
    return db.query(UserFeedback).filter(UserFeedback.user_id == user_id).order_by(UserFeedback.created_at.desc()).all()


def get_user_deadlines(db: Session, user_id: int) -> List[CaseDeadline]:
    """Get all active deadlines for a user"""
    now = dt.datetime.now(dt.timezone.utc)
    return db.query(CaseDeadline).filter(
        CaseDeadline.user_id == user_id,
        CaseDeadline.is_completed == False,
        CaseDeadline.deadline_date > now,
    ).order_by(CaseDeadline.deadline_date).all()


def get_case_documents(db: Session, case_id: int) -> List[CaseDocument]:
    """Get all documents for a case"""
    return db.query(CaseDocument).filter(
        CaseDocument.case_id == case_id
    ).order_by(CaseDocument.uploaded_at).all()


def get_case_document_by_id(db: Session, document_id: int) -> Optional[CaseDocument]:
    """Get a document by ID"""
    return db.query(CaseDocument).filter(CaseDocument.id == document_id).first()


def get_notification_template_for_user(db: Session, user_id: int) -> Optional[NotificationTemplate]:
    """Get notification template for a user"""
    return db.query(NotificationTemplate).filter(NotificationTemplate.user_id == user_id).first()


def _reserve_otp_rate_limit_slot(email: str, max_requests_per_hour: int) -> bool:
    """Internal helper for OTP rate limiting - mockable for tests"""
    # In the real app, this would use Redis or similar
    return True


def get_user_stats(db: Session, user_id: int) -> dict:
    """Calculate high-level stats for a user dashboard"""
    cases = get_user_cases(db, user_id)

    active_count = len([c for c in cases if c.status == CaseStatus.ACTIVE])
    appealed_count = len([c for c in cases if c.status == CaseStatus.APPEALED])
    closed_count = len([c for c in cases if c.status == CaseStatus.CLOSED])

    # Get upcoming deadlines count
    now = dt.datetime.now(dt.timezone.utc)
    upcoming_deadlines = db.query(CaseDeadline).filter(
        CaseDeadline.user_id == user_id,
        CaseDeadline.is_completed == False,
        CaseDeadline.deadline_date > now,
    ).count()

    return {
        "total_cases": len(cases),
        "active_cases": active_count,
        "appealed_cases": appealed_count,
        "closed_cases": closed_count,
        "upcoming_deadlines": upcoming_deadlines,
    }


def create_or_update_user_preference(
    db: Session,
    user_id: int,
    email: str,
    phone_number: Optional[str] = None,
    notification_channel: NotificationChannel = NotificationChannel.BOTH,
    timezone: str = "UTC",
    # Holiday-aware reminder engine (MVP)
    holiday_aware_reminders: bool = False,
    holiday_country: Optional[str] = None,
    holiday_region: Optional[str] = None,
    holiday_calendar_json: Optional[str] = None,
) -> UserPreference:
    """Create or update user notification preferences"""
    pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()

    if pref:
        pref.email = email
        pref.phone_number = phone_number
        pref.notification_channel = notification_channel
        pref.timezone = timezone
        pref.holiday_aware_reminders = holiday_aware_reminders
        pref.holiday_country = holiday_country
        pref.holiday_region = holiday_region
        pref.holiday_calendar_json = holiday_calendar_json
        pref.updated_at = dt.datetime.now(dt.timezone.utc)
    else:
        pref = UserPreference(
            user_id=user_id,
            email=email,
            phone_number=phone_number,
            notification_channel=notification_channel,
            timezone=timezone,
            holiday_aware_reminders=holiday_aware_reminders,
            holiday_country=holiday_country,
            holiday_region=holiday_region,
            holiday_calendar_json=holiday_calendar_json,
        )
        db.add(pref)
    
    db.commit()
    db.refresh(pref)
    return pref


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
    document_type: DocumentType,
    user_id: int,
    document_content: Optional[str] = None,
    file_path: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
) -> CaseDocument:
    """Create a new case document.

    Security: enforce that `case_id` belongs to `user_id` (server-side ownership
    validation), consistent with create_case_deadline.
    """
    try:
        normalized_case_id = int(case_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("case_id must be an integer matching cases.id") from exc

    # Ownership validation (prevents attaching documents to another user's case)
    case = db.query(Case).filter(Case.id == normalized_case_id).first()
    if not case or case.user_id != user_id:
        raise PermissionError(
            "case_id not found or not owned by the provided user_id"
        )

    doc = CaseDocument(
        case_id=normalized_case_id,
        document_type=document_type,
        document_content=document_content,
        file_path=file_path,
        summary=summary,
        remedies=remedies,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event
    db.refresh(doc)
    return doc


def get_case_documents(db: Session, case_id: int) -> List[CaseDocument]:
    """Get all documents for a case"""
    return db.query(CaseDocument).filter(
        CaseDocument.case_id == case_id
    ).order_by(CaseDocument.uploaded_at).all()


def get_case_document_by_id(db: Session, document_id: int) -> Optional[CaseDocument]:
    """Get a case document by ID"""
    return db.query(CaseDocument).filter(CaseDocument.id == document_id).first()


def update_case_document(
    db: Session,
    document_id: int,
    document_content: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
) -> Optional[CaseDocument]:
    """Update case document"""
    doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
    if doc:
        if document_content is not None:
            doc.document_content = document_content
        if summary is not None:
            doc.summary = summary
        if remedies is not None:
            doc.remedies = remedies
        try:
            db.commit()
            db.refresh(doc)
        except Exception as e:
            db.rollback()
            raise RuntimeError(f"Database write failed for case document {document_id}: {str(e)}") from e
    return doc


def create_attachment(
    db: Session,
    user_id: int,
    original_filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    case_id: Optional[int] = None,
    deadline_id: Optional[int] = None,
) -> Attachment:
    """Create a new file attachment record"""
    att = Attachment(
        user_id=user_id,
        original_filename=original_filename,
        stored_path=stored_path,
        content_type=content_type,
        size_bytes=size_bytes,
        case_id=case_id,
        deadline_id=deadline_id,
    )
    db.add(att)
    db.commit()
    db.refresh(att)
    return att


def get_attachments_for_case(db: Session, case_id: int) -> List[Attachment]:
    """Get all attachments for a case"""
    return db.query(Attachment).filter(Attachment.case_id == case_id).all()
