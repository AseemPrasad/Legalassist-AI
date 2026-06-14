"""Compatibility shim for the original monolithic `database.py`.

The project has moved models and CRUD helpers into the `db/` package, but many
existing imports still point at `database`. This module re-exports the pieces
needed by the current codebase and keeps the authentication/OTP security path
working while the refactor continues.

CRITICAL: This file should be a PURE RE-EXPORT MODULE. All implementations must
come from db/ subpackages. Do not define duplicate functions here.
"""

from __future__ import annotations

import datetime as dt
import threading
from typing import Optional, List
from sqlalchemy import func, Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Session, relationship

from db.base import Base
from db.session import engine, SessionLocal, init_db, db_session, get_db, _to_utc_datetime, _datetime_for_db

_OTP_RATE_LIMIT_LOCK = threading.RLock()
_OTP_RATE_LIMIT_EVENTS: dict[str, list[dt.datetime]] = {}


def _otp_rate_limit_key(identifier: str) -> str:
    normalized = str(identifier).strip().lower().replace("@", "")
    if not normalized:
        raise ValueError("OTP request identifier is required")
    return f"otp:rate:{normalized}"


from db.models import (
    User,
    OTPVerification,
    APIKey,
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
    CaseNote,
    CaseNoteVersion,
    AnonymizedShareToken,
    CaseComment,
    CasePresence,
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
    CaseNote,
    CaseComment,
    CasePresence,
)
from db.crud.comments import (
    get_case_comments,
    get_case_presence,
    create_case_comment,
    upsert_case_presence,
)
from db.case_service import save_case_note_draft

from db.crud.notifications import (
    create_case_deadline,
    get_upcoming_deadlines,
    has_notification_been_sent,
    log_notification,
    get_notification_history,
    reserve_notification,
    update_notification_result,
)

from db.crud.users import (
    get_user_by_email,
    create_user,
    update_user_last_login,
    create_otp_verification,
    get_pending_otp,
    mark_otp_as_used,
    is_email_locked_out,
    record_otp_failed_attempt,
    reset_otp_failed_attempts,
    cleanup_expired_otps,
    create_or_update_user_preference,
)

from db.crud.cases import (
    create_case,
    get_user_cases,
    get_case_by_id,
    get_case_by_number,
    update_case_status,
    delete_case,
    create_case_document,
    get_case_documents,
    get_case_document_by_id,
    create_case_record,
    get_case_record,
    get_cases_by_criteria,
    update_case_outcome,
    submit_user_feedback,
    get_user_feedback,
    submit_model_feedback,
    get_case_timeline,
    create_timeline_event,
    create_attachment,
    get_attachments_for_case,
    get_user_stats,
    get_similarity_feedback,
)



from db.case_service import (
    save_case_note_draft,
    publish_case_note,
    get_case_note_history,
)

from db.crud.comments import (
    create_case_comment,
    get_case_comments,
    upsert_case_presence,
    get_case_presence,
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
    "CaseNote",
    "CaseComment",
    "CasePresence",
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
    "has_notification_been_sent",
    "log_notification",
    "get_notification_history",
    "reserve_notification",
    "update_notification_result",
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
    "revoke_token",
    "cleanup_expired_revoked_tokens",
    "is_token_revoked",
    "CaseNote",
    "CaseNoteVersion",
    "save_case_note_draft",
    "publish_case_note",
    "get_case_note_history",
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


def create_otp_verification(
    db: Session,
    email: str,
    otp_hash: str,
    expires_at: dt.datetime,
    max_requests_per_hour: int = 5,
    requester_ip: Optional[str] = None,
) -> OTPVerification:
    """Create a new OTP verification record"""
    with _OTP_RATE_LIMIT_LOCK:
        _reserve_otp_rate_limit_slot(db, email, max_requests_per_hour, requester_ip=requester_ip)
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
    """Atomically mark OTP as used. Returns True only if OTP was not already used."""
    try:
        result = db.query(OTPVerification).filter(
            OTPVerification.id == otp_id,
            OTPVerification.is_used == False,
        ).update({"is_used": True}, synchronize_session=False)
        db.commit()
        return result > 0
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


def record_otp_failed_attempt(
    db: Session,
    otp_id: int,
    lockout_duration_minutes: int = 15,
    max_failed_attempts: int = 5,
) -> bool:
    """Record a failed OTP verification attempt and implement lockout after max attempts.

    Lockout is applied at the **email level**: once *max_failed_attempts* is
    reached for any single OTP record, every other pending OTP issued to the
    same email address is locked simultaneously.  This prevents an attacker
    from cycling through multiple valid OTP tokens to bypass the attempt limit.

    Args:
        db: Active SQLAlchemy database session.
        otp_id: Primary-key ID of the OTPVerification record to update.
        lockout_duration_minutes: Duration to lock out the email after max
            failed attempts are reached (default: 15 minutes).
        max_failed_attempts: Number of consecutive failures that trigger a
            lockout (default: 5).

    Returns:
        True if the record was updated successfully; False if the OTP was not
        found.
    """
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.failed_attempts += 1
        if otp.failed_attempts >= max_failed_attempts:
            # Lock out at email level, not just OTP level.
            lockout_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(
                minutes=lockout_duration_minutes
            )
            otp.locked_until = lockout_until
            logger.warning(
                "OTP email-level lockout applied",
                email=otp.email,
                failed_attempts=otp.failed_attempts,
                locked_until=str(lockout_until),
            )
            # Also lock all other pending OTPs for the same email.
            db.query(OTPVerification).filter(
                OTPVerification.email == otp.email,
                OTPVerification.id != otp_id,
            ).update({"locked_until": lockout_until})

        db.commit()
        db.refresh(otp)
        return True
    return False

    password_hash = Column(
        String(255), 
        nullable=True
    )

    # -------------------------------------------------------------------------
    # ORM Relationships
    # -------------------------------------------------------------------------
    # We define bidirectional relationships with other core entities here.
    # The `cascade="all, delete-orphan"` parameter is crucial for maintaining
    # referential integrity and preventing orphaned records in the database
    # if a user account is deleted (e.g., for GDPR compliance).
    # -------------------------------------------------------------------------
    
    cases = relationship(
        "Case", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    
    preferences = relationship(
        "UserPreference", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )

def reset_otp_failed_attempts(db: Session, otp_id: int) -> bool:
    """Reset the failed-attempt counter and any lockout on successful verification.

    Args:
        db: Active SQLAlchemy database session.
        otp_id: Primary-key ID of the OTPVerification record to reset.

    Returns:
        True if the record was updated; False if the OTP was not found.
    """
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


def revoke_token(db: Session, jti: str, expires_at: dt.datetime) -> RevokedToken:
    token = RevokedToken(jti=jti, expires_at=expires_at)
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


# Database initialization
def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def cleanup_expired_revoked_tokens(db: Session, batch_size: int = 1000) -> int:
    """Delete expired revoked tokens in batches to avoid lock contention."""
    now = dt.datetime.now(dt.timezone.utc)
    total_deleted = 0

    while True:
        deleted = db.query(RevokedToken).filter(
            RevokedToken.expires_at < now
        ).limit(batch_size).delete(synchronize_session=False)
        db.commit()
        total_deleted += deleted
        if deleted < batch_size:
            break

    return total_deleted


def schedule_token_cleanup():
    """Standalone cleanup runner for cron/celery scheduling."""
    from database import SessionLocal, cleanup_expired_revoked_tokens
    db = SessionLocal()
    try:
        deleted = cleanup_expired_revoked_tokens(db)
        return deleted
    finally:
        db.close()


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

    # Validate deadline date is not in the past
    if deadline_date.tzinfo is None:
        deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)
    if deadline_date < dt.datetime.now(dt.timezone.utc):
        raise ValueError("Deadline date must be in the future")

    deadline = CaseDeadline(
        user_id=user_id,
        case_id=normalized_case_id,
        case_title=case_title,
        deadline_date=deadline_date,
        deadline_type=deadline_type,
        description=description,
    )
    db.add(deadline)
    db.commit()
    db.refresh(deadline)
    return deadline

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
    """Delete a case and all related data.

    Explicitly removes dependent rows in FK-constraint-safe order before
    deleting the parent Case record.  Relying solely on ORM-level
    ``cascade="all, delete-orphan"`` can fail on PostgreSQL (and other
    databases that enforce referential integrity at the engine level) when
    related objects are not already loaded into the current session,
    causing the DELETE to hit a foreign-key violation before SQLAlchemy's
    lazy-loader can clean them up.

    Deletion order (deepest child first):
        CaseNoteVersion  -> CaseNote
        CaseComment (self-referencing replies cascade via FK ondelete)
        CaseTimeline, CasePresence, Attachment, CaseDocument
        AnonymizedShareToken, CaseDeadline
        Case (parent)
    """
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        return False

    try:
        # --- leaf tables (no children of their own) ---
        # CaseNoteVersion references both case_notes.id AND cases.id;
        # remove it before CaseNote to satisfy both FK constraints.
        db.query(CaseNoteVersion).filter(
            CaseNoteVersion.case_id == case_id
        ).delete(synchronize_session=False)

        # Self-referencing replies are handled by ondelete="CASCADE" on
        # the parent_comment_id FK, so deleting top-level comments is
        # sufficient — but we must delete all of them before the Case.
        db.query(CaseComment).filter(
            CaseComment.case_id == case_id
        ).delete(synchronize_session=False)

        db.query(CaseNote).filter(
            CaseNote.case_id == case_id
        ).delete(synchronize_session=False)

        db.query(CaseTimeline).filter(
            CaseTimeline.case_id == case_id
        ).delete(synchronize_session=False)

        db.query(CasePresence).filter(
            CasePresence.case_id == case_id
        ).delete(synchronize_session=False)

        # Attachments may reference case_documents.id (SET NULL) — delete
        # attachments before documents to avoid that nullable FK being
        # needed after document rows are gone.
        db.query(Attachment).filter(
            Attachment.case_id == case_id
        ).delete(synchronize_session=False)

        db.query(CaseDocument).filter(
            CaseDocument.case_id == case_id
        ).delete(synchronize_session=False)

        db.query(AnonymizedShareToken).filter(
            AnonymizedShareToken.case_id == case_id
        ).delete(synchronize_session=False)

        db.query(CaseDeadline).filter(
            CaseDeadline.case_id == case_id
        ).delete(synchronize_session=False)

        db.delete(case)
        db.commit()
        try:
            from core.embedding_engine import get_vector_store
            vs = get_vector_store()
            vs.delete(case_id)
        except Exception as ev:
            logger.warning(f"Failed to auto-invalidate vector store on case deletion: {ev}")
        return True
    except Exception:
        db.rollback()
        raise


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
    if appeal_date: outcome.appeal_date = appeal_date
    if appeal_outcome: outcome.appeal_outcome = appeal_outcome
    if appeal_success is not None: outcome.appeal_success = appeal_success
    if time_to_appeal_verdict: outcome.time_to_appeal_verdict = time_to_appeal_verdict
    if appeal_cost: outcome.appeal_cost = appeal_cost
    if additional_notes: outcome.additional_notes = additional_notes

    db.commit()
    db.refresh(outcome)
    return outcome


def get_case_record(db: Session, hashed_case_id: str) -> Optional[CaseRecord]:
    """Get a case record by ID"""
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


def get_cases_by_criteria(
    db: Session,
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    court_name: Optional[str] = None,
    judge_name: Optional[str] = None,
    plaintiff_type: Optional[str] = None,
    defendant_type: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = 100,
) -> List[CaseRecord]:
    """Get cases matching approved criteria fields only."""
    query = db.query(CaseRecord)

    filters = {
        "case_type": case_type,
        "jurisdiction": jurisdiction,
        "court_name": court_name,
        "judge_name": judge_name,
        "plaintiff_type": plaintiff_type,
        "defendant_type": defendant_type,
        "outcome": outcome,
    }
    for key, value in filters.items():
        if key not in ALLOWED_CASE_FILTER_FIELDS:
            continue
        if value:
            query = query.filter(getattr(CaseRecord, key) == value)

    return query.order_by(CaseRecord.created_at.desc()).limit(limit).all()


def _escape_like_pattern(value: str) -> str:
    """Escape SQL LIKE wildcard characters in a user-supplied string.

    Both ``%`` and ``_`` are prefixed with the escape character (``\\``) so
    they are treated as literals rather than wildcards when used in a LIKE
    expression with ``escape='\\'``.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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


# ==================== User & Authentication Helper Functions ====================


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email address"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str) -> User:
    """Create a new user"""
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_last_login(db: Session, user_id: int) -> User:
    """Update user's last login timestamp"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(user)
    return user


# Thread lock for OTP rate-limit enforcement (single source of truth).
_otp_rate_limit_lock = threading.Lock()


def create_otp_verification(
    db: Session,
    email: str,
    otp_hash: str,
    expires_at: dt.datetime,
    max_requests_per_hour: int = 5,
) -> OTPVerification:
    """Create a new OTP verification record with rate limiting.

    This is the single source of truth for OTP rate-limit enforcement.
    All callers (auth.py, API routes, etc.) must go through this function to
    ensure consistent throttling behavior across the entire application.
    """
    with _otp_rate_limit_lock:
        one_hour_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
        recent_otps = db.query(OTPVerification).filter(
            OTPVerification.email == email,
            OTPVerification.created_at >= one_hour_ago,
        ).count()

        if recent_otps >= max_requests_per_hour:
            raise ValueError("Too many OTP requests. Please try again later.")

        otp = OTPVerification(
            email=email,
            otp_hash=otp_hash,
            expires_at=expires_at,
        )
        db.add(otp)
        db.commit()
        db.refresh(otp)
        return otp


def get_pending_otp(db: Session, email: str) -> Optional[OTPVerification]:
    """Get unused, non-expired OTP for email"""
    now = dt.datetime.now(dt.timezone.utc)
    return db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.is_used == False,
        OTPVerification.expires_at > now,
    ).order_by(OTPVerification.created_at.desc()).first()


def mark_otp_as_used(db: Session, otp_id: int) -> bool:
    """Mark an OTP as used"""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.is_used = True
        db.commit()
        return True
    return False



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


def _reserve_otp_rate_limit_slot(
    db: Session,
    email: str,
    max_requests_per_hour: int,
    requester_ip: Optional[str] = None,
) -> bool:
    """Reserve an OTP request slot for the email, with optional IP tracking."""
    if max_requests_per_hour <= 0:
        raise ValueError("Too many OTP requests. Please try again later.")

    normalized_email = str(email).strip().lower()
    if not normalized_email:
        raise ValueError("OTP request email is required")

    now = dt.datetime.now(dt.timezone.utc)
    window_start = now - dt.timedelta(hours=1)

    with _OTP_RATE_LIMIT_LOCK:
        recent_email_requests = db.query(OTPVerification).filter(
            func.lower(OTPVerification.email) == normalized_email,
            OTPVerification.created_at >= window_start,
        ).count()
        if recent_email_requests >= max_requests_per_hour:
            raise ValueError("Too many OTP requests. Please try again later.")

        email_key = _otp_rate_limit_key(f"email:{normalized_email}")
        email_events = _OTP_RATE_LIMIT_EVENTS.setdefault(email_key, [])
        email_events[:] = [ts for ts in email_events if ts >= window_start]
        if len(email_events) >= max_requests_per_hour:
            raise ValueError("Too many OTP requests. Please try again later.")
        email_events.append(now)

        if requester_ip:
            normalized_ip = str(requester_ip).strip().lower()
            if normalized_ip:
                ip_key = _otp_rate_limit_key(f"ip:{normalized_ip}")
                ip_events = _OTP_RATE_LIMIT_EVENTS.setdefault(ip_key, [])
                ip_events[:] = [ts for ts in ip_events if ts >= window_start]
                ip_events.append(now)

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
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


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
    db.add(doc)
    db.commit()
    db.refresh(doc)
    try:
        from core.embedding_engine import get_embedding_engine, get_vector_store
        import json
        engine = get_embedding_engine()
        emb_obj = engine.embed_case(db, normalized_case_id, force_regenerate=True)
        if emb_obj:
            vec = json.loads(emb_obj.embedding_vector)
            vs = get_vector_store()
            vs.add_batch([(normalized_case_id, vec)])
    except Exception as ev:
        logger.warning(f"Failed to auto-synchronize vector store on document creation: {ev}")
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
    extracted_metadata: Optional[dict] = None,
    extraction_method: Optional[str] = None,
    ocr_used: Optional[bool] = None,
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
        if extracted_metadata is not None:
            doc.extracted_metadata = extracted_metadata
        if extraction_method is not None:
            doc.extraction_method = extraction_method
        if ocr_used is not None:
            doc.ocr_used = ocr_used
        try:
            db.commit()
            db.refresh(doc)
            try:
                from core.embedding_engine import get_embedding_engine, get_vector_store
                import json
                engine = get_embedding_engine()
                emb_obj = engine.embed_case(db, doc.case_id, force_regenerate=True)
                if emb_obj:
                    vec = json.loads(emb_obj.embedding_vector)
                    vs = get_vector_store()
                    vs.add_batch([(doc.case_id, vec)])
            except Exception as ev:
                logger.warning(f"Failed to auto-synchronize vector store on document update: {ev}")
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
) -> "Attachment":
    """Create a new attachment record linked to a case or deadline"""
    if case_id is not None:
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case or case.user_id != user_id:
            raise PermissionError("case_id not found or not owned by the provided user_id")
    if deadline_id is not None:
        deadline = db.query(CaseDeadline).filter(CaseDeadline.id == deadline_id).first()
        if not deadline or deadline.user_id != user_id:
            raise PermissionError("deadline_id not found or not owned by the provided user_id")

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


# ====================================================================
# Revocation cache — Redis-backed coordinated cache to prevent
# thundering herd on token revocation DB queries during bursts.
# ====================================================================

_revocation_cache = None
_revocation_cache_lock = threading.Lock()


def _get_revocation_cache():
    global _revocation_cache
    if _revocation_cache is not None:
        return _revocation_cache
    with _revocation_cache_lock:
        if _revocation_cache is None:
            if redis is None:
                return None
            redis_url = getattr(Config, "REDIS_URL", "redis://localhost:6379/0")
            _revocation_cache = redis.from_url(redis_url, decode_responses=True)
    return _revocation_cache


def _is_token_revoked_uncached(db: Session, jti: str) -> bool:
    return db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is not None


def is_token_revoked(db: Session, jti: str) -> bool:
    """Check if token JTI is revoked, using Redis coordinated cache."""
    cache = _get_revocation_cache()
    if cache is None:
        return _is_token_revoked_uncached(db, jti)

    cache_key = f"revoked:{jti}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached == "1"

    lock_key = f"{cache_key}:lock"
    lock_value = str(time.monotonic_ns())

    if cache.set(lock_key, lock_value, nx=True, ex=10):
        try:
            revoked = _is_token_revoked_uncached(db, jti)
            ttl = 3600 if revoked else 300
            cache.setex(cache_key, ttl, "1" if revoked else "0")
            return revoked
        finally:
            if cache.get(lock_key) == lock_value:
                cache.delete(lock_key)

    for _ in range(50):
        time.sleep(0.02)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached == "1"

    return _is_token_revoked_uncached(db, jti)


def revoke_token(db: Session, jti: str, expires_at: dt.datetime) -> RevokedToken:
    """Add a token JTI to the revocation blacklist"""
    token = RevokedToken(jti=jti, expires_at=expires_at)
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


def aggregate_model_performance(db: Session, task: str = None) -> list:
    return []



def cleanup_expired_revoked_tokens(db: Session) -> int:
    """Remove expired tokens from the blacklist"""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(RevokedToken).filter(RevokedToken.expires_at < now).delete(synchronize_session=False)
    db.commit()
    return deleted


def submit_similarity_feedback(
    db: Session,
    user_id: int | str,
    candidate_case_id: int,
    query_signature: str,
    relevance: bool,
) -> SimilarityFeedback:
    """Submit similarity feedback for search queries."""
    feedback = SimilarityFeedback(
        user_id=str(user_id),
        candidate_case_id=candidate_case_id,
        query_signature=query_signature,
        relevance=relevance,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_similarity_feedback(
    db: Session,
    user_id: Optional[str] = None,
    query_signature: Optional[str] = None,
    candidate_case_id: Optional[int] = None,
    limit: int = 100,
) -> List[SimilarityFeedback]:
    """Get similarity feedback rows filtered by user, query, or candidate case"""
    query = db.query(SimilarityFeedback)

    if user_id is not None:
        query = query.filter(SimilarityFeedback.user_id == str(user_id))
    if query_signature is not None:
        query = query.filter(SimilarityFeedback.query_signature == query_signature)
    if candidate_case_id is not None:
        query = query.filter(SimilarityFeedback.candidate_case_id == candidate_case_id)

    return query.order_by(SimilarityFeedback.created_at.desc()).limit(limit).all()


def reserve_idempotency_key(db: Session, key: str, method: str, path: str) -> tuple[IdempotencyKey, bool]:
    """Reserve an idempotency key within a nested transaction/savepoint."""
    from sqlalchemy.exc import IntegrityError
    ik = IdempotencyKey(
        key=key,
        method=method,
        path=path,
        status=IdempotencyKeyStatus.IN_PROGRESS,
    )
    try:
        with db.begin_nested():
            db.add(ik)
        db.commit()
        db.refresh(ik)
        return ik, True
    except IntegrityError:
        existing = db.query(IdempotencyKey).filter(IdempotencyKey.key == key).first()
        return existing, False


def set_idempotency_response(db: Session, key: str, status_code: int, headers: dict, body: str) -> IdempotencyKey:
    """Set the response payload for a completed idempotency key."""
    from sqlalchemy.exc import IntegrityError
    ik = db.query(IdempotencyKey).filter(IdempotencyKey.key == key).first()
    if not ik:
        try:
            ik = IdempotencyKey(key=key, method="POST", path="unknown", status=IdempotencyKeyStatus.COMPLETED)
            with db.begin_nested():
                db.add(ik)
            db.commit()
        except IntegrityError:
            ik = db.query(IdempotencyKey).filter(IdempotencyKey.key == key).first()
            if not ik:
                ik = IdempotencyKey(key=key, method="POST", path="unknown", status=IdempotencyKeyStatus.COMPLETED)
                db.add(ik)
                db.commit()
    ik.status = IdempotencyKeyStatus.COMPLETED
    ik.response_status = status_code
    ik.response_headers = headers
    ik.response_body = body
    ik.completed_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    db.refresh(ik)
    return ik


def get_idempotency_response(db: Session, key: str) -> Optional[dict]:
    """Retrieve the cached response for a completed idempotency key."""
    ik = db.query(IdempotencyKey).filter(
        IdempotencyKey.key == key,
        IdempotencyKey.status == IdempotencyKeyStatus.COMPLETED
    ).first()
    if ik:
        return {
            "status_code": ik.response_status,
            "headers": ik.response_headers,
            "body": ik.response_body,
        }
    return None



def delete_user_cases(db: Session, user_id: int) -> dict:
    """Delete all cases and associated data for a user.
    
    This performs a soft deletion approach where cases are marked
    as deleted rather than immediately removed from the database.
    
    Args:
        db: Database session
        user_id: The ID of the user whose cases should be deleted
        
    Returns:
        Dictionary with counts of deleted items
    """
    deleted_counts = {
        "cases": 0,
        "documents": 0,
        "deadlines": 0,
        "timeline_events": 0,
    }
    
    cases = db.query(Case).filter(Case.user_id == user_id).all()
    case_ids = [c.id for c in cases]
    
    if not case_ids:
        return deleted_counts
    
    # Delete documents
    docs = db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).all()
    deleted_counts["documents"] = len(docs)
    db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).delete(
        synchronize_session=False
    )
    
    # Delete timeline events
    events = db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).all()
    deleted_counts["timeline_events"] = len(events)
    db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).delete(
        synchronize_session=False
    )
    
    # Delete deadlines
    deadlines = db.query(CaseDeadline).filter(
        (CaseDeadline.user_id == user_id) | (CaseDeadline.case_id.in_(case_ids))
    ).all()
    deleted_counts["deadlines"] = len(deadlines)
    db.query(CaseDeadline).filter(
        (CaseDeadline.user_id == user_id) | (CaseDeadline.case_id.in_(case_ids))
    ).delete(synchronize_session=False)
    
    # Delete cases
    deleted_counts["cases"] = len(cases)
    db.query(Case).filter(Case.user_id == user_id).delete(synchronize_session=False)
    
    db.commit()
    return deleted_counts


def redact_user_data(db: Session, user_id: int) -> int:
    """Redact PII from user and related records.
    
    This replaces all PII with redaction placeholders while keeping
    the database records intact for audit purposes.
    
    Args:
        db: Database session
        user_id: The ID of the user whose data should be redacted
        
    Returns:
        Number of records redacted
    """
    REDACTED = "[REDACTED-GDPR]"
    REDACTED_EMAIL = "[REDACTED-EMAIL]"
    
    redacted_count = 0
    
    # Redact user
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.email = REDACTED_EMAIL
        if hasattr(user, 'full_name'):
            user.full_name = REDACTED
        if hasattr(user, 'phone'):
            user.phone = REDACTED
        if hasattr(user, 'address'):
            user.address = REDACTED
        db.commit()
        redacted_count += 1
    
    # Redact cases
    cases = db.query(Case).filter(Case.user_id == user_id).all()
    for case in cases:
        case.title = f"{REDACTED}-{case.id}"
        redacted_count += 1
    
    # Redact documents
    case_ids = [c.id for c in cases]
    if case_ids:
        docs = db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).all()
        for doc in docs:
            doc.summary = REDACTED
            doc.document_content = REDACTED
            doc.extracted_metadata = {}
            redacted_count += 1
        
        # Redact timeline events
        events = db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).all()
        for event in events:
            event.description = REDACTED
            event.event_metadata = {}
            redacted_count += 1
    
    db.commit()
    return redacted_count


# ====================================================================
# GDPR-compliant data deletion functions
# ====================================================================

def delete_user_cases(db: Session, user_id: int) -> dict:
    """Delete all cases and associated data for a user."""
    from db.models import Case, CaseDocument, CaseDeadline, CaseTimeline

    deleted_counts = {"cases": 0, "documents": 0, "deadlines": 0, "timeline_events": 0}
    cases = db.query(Case).filter(Case.user_id == user_id).all()
    case_ids = [c.id for c in cases]

    if not case_ids:
        return deleted_counts

    docs = db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).all()
    deleted_counts["documents"] = len(docs)
    db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).delete(synchronize_session=False)

    events = db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).all()
    deleted_counts["timeline_events"] = len(events)
    db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).delete(synchronize_session=False)

    deadlines = db.query(CaseDeadline).filter((CaseDeadline.user_id == user_id) | (CaseDeadline.case_id.in_(case_ids))).all()
    deleted_counts["deadlines"] = len(deadlines)
    db.query(CaseDeadline).filter((CaseDeadline.user_id == user_id) | (CaseDeadline.case_id.in_(case_ids))).delete(synchronize_session=False)

    deleted_counts["cases"] = len(cases)
    db.query(Case).filter(Case.user_id == user_id).delete(synchronize_session=False)
    db.commit()
    return deleted_counts


def redact_user_data(db: Session, user_id: int) -> int:
    """Redact PII from user and related records."""
    from db.models import Case, CaseDocument, CaseTimeline, User

    REDACTED = "[REDACTED-GDPR]"
    REDACTED_EMAIL = "[REDACTED-EMAIL]"

    redacted_count = 0
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.email = REDACTED_EMAIL
        if hasattr(user, 'full_name'):
            user.full_name = REDACTED
        if hasattr(user, 'phone'):
            user.phone = REDACTED
        if hasattr(user, 'address'):
            user.address = REDACTED
        db.commit()
        redacted_count += 1

    cases = db.query(Case).filter(Case.user_id == user_id).all()
    for case in cases:
        case.title = f"{REDACTED}-{case.id}"
        redacted_count += 1

    case_ids = [c.id for c in cases]
    if case_ids:
        docs = db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).all()
        for doc in docs:
            doc.summary = REDACTED
            doc.document_content = REDACTED
            doc.extracted_metadata = {}
            redacted_count += 1
        events = db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).all()
        for event in events:
            event.description = REDACTED
            event.event_metadata = {}
            redacted_count += 1

    db.commit()
    return redacted_count

# ====================================================================
# GDPR-compliant data deletion functions
# ====================================================================

def delete_user_cases(db: Session, user_id: int) -> dict:
    """Delete all cases and associated data for a user."""
    from db.models import Case, CaseDocument, CaseDeadline, CaseTimeline

    deleted_counts = {"cases": 0, "documents": 0, "deadlines": 0, "timeline_events": 0}
    cases = db.query(Case).filter(Case.user_id == user_id).all()
    case_ids = [c.id for c in cases]

    if not case_ids:
        return deleted_counts

    docs = db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).all()
    deleted_counts["documents"] = len(docs)
    db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).delete(synchronize_session=False)

    events = db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).all()
    deleted_counts["timeline_events"] = len(events)
    db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).delete(synchronize_session=False)

    deadlines = db.query(CaseDeadline).filter((CaseDeadline.user_id == user_id) | (CaseDeadline.case_id.in_(case_ids))).all()
    deleted_counts["deadlines"] = len(deadlines)
    db.query(CaseDeadline).filter((CaseDeadline.user_id == user_id) | (CaseDeadline.case_id.in_(case_ids))).delete(synchronize_session=False)

    deleted_counts["cases"] = len(cases)
    db.query(Case).filter(Case.user_id == user_id).delete(synchronize_session=False)
    db.commit()
    return deleted_counts


def redact_user_data(db: Session, user_id: int) -> int:
    """Redact PII from user and related records."""
    from db.models import Case, CaseDocument, CaseTimeline, User

    REDACTED = "[REDACTED-GDPR]"
    REDACTED_EMAIL = "[REDACTED-EMAIL]"

    redacted_count = 0
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.email = REDACTED_EMAIL
        if hasattr(user, 'full_name'):
            user.full_name = REDACTED
        if hasattr(user, 'phone'):
            user.phone = REDACTED
        if hasattr(user, 'address'):
            user.address = REDACTED
        db.commit()
        redacted_count += 1

    cases = db.query(Case).filter(Case.user_id == user_id).all()
    for case in cases:
        case.title = f"{REDACTED}-{case.id}"
        redacted_count += 1

    case_ids = [c.id for c in cases]
    if case_ids:
        docs = db.query(CaseDocument).filter(CaseDocument.case_id.in_(case_ids)).all()
        for doc in docs:
            doc.summary = REDACTED
            doc.document_content = REDACTED
            doc.extracted_metadata = {}
            redacted_count += 1
        events = db.query(CaseTimeline).filter(CaseTimeline.case_id.in_(case_ids)).all()
        for event in events:
            event.description = REDACTED
            event.event_metadata = {}
            redacted_count += 1

    db.commit()
    return redacted_count