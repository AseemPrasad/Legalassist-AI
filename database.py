"""Compatibility shim for the original monolithic `database.py`.

The project now keeps business logic in `db/` modules and `db.crud.*` helpers.
This file remains as a stable public API surface for legacy imports.
"""

from __future__ import annotations

from db.attachments_service import create_attachment, get_attachments_for_case
import datetime as dt
import hashlib
import threading
from config import Config
try:
    import redis
except ImportError:
    redis = None

from typing import Optional, List
from sqlalchemy import func
from sqlalchemy.orm import Session

from db.base import Base
from db.case_service import (
    create_case,
    create_case_document,
    create_case_record,
    create_timeline_event,
    delete_case,
    get_case_by_id,
    get_case_by_number,
    get_case_document_by_id,
    get_case_documents,
    get_case_record,
    get_case_timeline,
    get_cases_by_criteria,
    get_user_cases,
    get_user_stats,
    submit_model_feedback,
    update_case_document,
    update_case_outcome,
    update_case_status,
)
from db.crud.feedback import get_user_feedback, submit_user_feedback
from db.crud.notifications import (
    create_case_deadline,
    get_notification_history,
    get_notification_template_for_user,
    get_upcoming_deadlines,
    has_notification_been_sent,
    log_notification,
)
from db.models import (
    Attachment,
    Case,
    CaseAnalytics,
    CaseArgument,
    CaseDeadline,
    CaseDocument,
    CaseEmbedding,
    CaseIssue,
    CaseOutcome,
    CaseRecord,
    CaseStatus,
    CaseTimeline,
    DocumentType,
    KnowledgeGraphEdge,
    ModelFeedback,
    ModelPerformance,
    ModelRoutingRule,
    NotificationChannel,
    NotificationLog,
    NotificationStatus,
    NotificationTemplate,
    OTPVerification,
    PrecedentMatch,
    Report,
)
from db.notifications_service import create_or_update_user_preference, get_user_deadlines
from db.otp_service import (
    _get_otp_rate_limit_script,
    _otp_rate_limit_key,
    _reserve_otp_rate_limit_slot,
    cleanup_expired_otps,
    cleanup_expired_revoked_tokens,
    create_otp_verification,
    create_user,
    get_pending_otp,
    record_otp_failed_attempt,
    get_user_by_email,
    is_token_revoked,
    mark_otp_as_used,
    revoke_token,
    reset_otp_failed_attempts,
    update_user_last_login,
)
from db.session import db_session, engine, get_db, init_db, SessionLocal, _datetime_for_db, _to_utc_datetime

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
    "Report",
    "create_case_deadline",
    "get_upcoming_deadlines",
    "get_prefs_by_user_ids",
    "get_user_deadlines",
    "has_notification_been_sent",
    "log_notification",
    "get_notification_history",
    "get_notification_template_for_user",
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
    "record_otp_failed_attempt",
    "reset_otp_failed_attempts",
    "revoke_token",
    "is_token_revoked",
    "cleanup_expired_revoked_tokens",
    "create_case",
    "get_user_cases",
    "get_case_by_id",
    "get_case_by_number",
    "update_case_status",
    "delete_case",
    "get_user_stats",
    "create_case_document",
    "get_case_documents",
    "get_case_document_by_id",
    "update_case_document",
    "create_case_record",
    "get_case_record",
    "get_cases_by_criteria",
    "update_case_outcome",
    "get_user_feedback",
    "submit_user_feedback",
    "submit_model_feedback",
    "get_case_timeline",
    "create_timeline_event",
    "create_attachment",
    "get_attachments_for_case",
    "submit_similarity_feedback",
    "get_similarity_feedback",
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


_OTP_RATE_LIMIT_WINDOW_SECONDS = 60 * 60
_OTP_RATE_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""
_otp_rate_limit_client = None
_otp_rate_limit_script = None
_otp_rate_limit_lock = threading.Lock()


def _otp_rate_limit_key(identifier: str) -> str:
    normalized = str(identifier).strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"otp:rate:{digest}"


def _get_otp_rate_limit_script():
    global _otp_rate_limit_client, _otp_rate_limit_script

    if _otp_rate_limit_script is not None:
        return _otp_rate_limit_script

    with _otp_rate_limit_lock:
        if _otp_rate_limit_script is None:
            if redis is None:
                raise RuntimeError("Redis is required for OTP rate limiting but is not installed.")

            redis_url = getattr(Config, "REDIS_URL", "redis://localhost:6379/0")
            _otp_rate_limit_client = redis.from_url(redis_url, decode_responses=True)
            _otp_rate_limit_script = _otp_rate_limit_client.register_script(_OTP_RATE_LIMIT_SCRIPT)

    return _otp_rate_limit_script


def _reserve_otp_rate_limit_slot(identifier: str, max_requests_per_hour: int, label: str = "identifier") -> int:
    normalized_identifier = str(identifier).strip().lower()
    if not normalized_identifier:
        raise ValueError(f"{label} is required for OTP rate limiting")

    script = _get_otp_rate_limit_script()
    current = int(script(keys=[_otp_rate_limit_key(normalized_identifier)], args=[_OTP_RATE_LIMIT_WINDOW_SECONDS]))

    if current > max_requests_per_hour:
        raise ValueError("Too many OTP requests. Please try again later.")

    return current


def _safe_reserve_otp_slot(identifier: str, max_requests_per_hour: int, label: str = "identifier") -> int:
    try:
        return _reserve_otp_rate_limit_slot(identifier, max_requests_per_hour, label=label)
    except TypeError:
        try:
            return _reserve_otp_rate_limit_slot(identifier, max_requests_per_hour, label)
        except TypeError:
            return _reserve_otp_rate_limit_slot(identifier, max_requests_per_hour)


def create_otp_verification(
    db: Session,
    email: str,
    otp_hash: str,
    expires_at: dt.datetime,
    max_requests_per_hour: int = 5,
    requester_ip: Optional[str] = None,
) -> OTPVerification:
    """Create a new OTP verification record with rate limiting."""
    _safe_reserve_otp_slot(email, max_requests_per_hour, label="Email")
    if requester_ip:
        _safe_reserve_otp_slot(requester_ip, max_requests_per_hour, label="IP")

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


def record_otp_failed_attempt(db: Session, otp_id: int, lockout_duration_minutes: int = 15, max_failed_attempts: int = 5) -> bool:
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.failed_attempts += 1
        if otp.failed_attempts >= max_failed_attempts:
            otp.locked_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=lockout_duration_minutes)
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


def revoke_token(db: Session, jti: str, expires_at: dt.datetime) -> RevokedToken:
    token = RevokedToken(jti=jti, expires_at=expires_at)
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


def is_token_revoked(db: Session, jti: str) -> bool:
    return db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is not None


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
    db.add(doc)
    db.commit()
    db.refresh(event)
    return event


def create_case_document_secure(
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
        db.commit()
        db.refresh(doc)
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


def is_token_revoked(db: Session, jti: str) -> bool:
    """Check if token JTI is in the revocation blacklist"""
    return db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is not None


def revoke_token(db: Session, jti: str, expires_at: dt.datetime) -> RevokedToken:
    """Add a token JTI to the revocation blacklist"""
    token = RevokedToken(jti=jti, expires_at=expires_at)
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


def cleanup_expired_revoked_tokens(db: Session) -> int:
    """Remove expired tokens from the blacklist"""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(RevokedToken).filter(RevokedToken.expires_at < now).delete(synchronize_session=False)
    db.commit()
    return deleted


def submit_similarity_feedback(
    db: Session,
    user_id: str,
    candidate_case_id: int,
    query_signature: str,
    relevance: bool,
) -> SimilarityFeedback:
    """Persist feedback for a similarity search result"""
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


def record_otp_failed_attempt(
    db: Session,
    otp_id: int,
    lockout_duration_minutes: int = 15,
    max_failed_attempts: int = 5,
) -> OTPVerification:
    """Record a failed OTP verification attempt and lock if max reached"""
    otp_record = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp_record:
        otp_record.failed_attempts += 1
        if otp_record.failed_attempts >= max_failed_attempts:
            otp_record.locked_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=lockout_duration_minutes)
        db.commit()
        db.refresh(otp_record)
    return otp_record


def reset_otp_failed_attempts(db: Session, otp_id: int) -> OTPVerification:
    """Reset OTP failed attempts counter and unlock"""
    otp_record = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp_record:
        otp_record.failed_attempts = 0
        otp_record.locked_until = None
        db.commit()
        db.refresh(otp_record)
    return otp_record



