import datetime as dt
from typing import Optional, List, Iterable
from sqlalchemy.orm import Session
from db.models.notifications import NotificationLog, NotificationStatus, NotificationChannel, NotificationTemplate, UserPreference
from db.models.cases import CaseDeadline, Case
from sqlalchemy.exc import IntegrityError


def get_or_create_notification_log(
    db: Session,
    deadline_id: int,
    user_id: int,
    channel: NotificationChannel,
    recipient: str,
    days_before: int,
) -> NotificationLog:
    """Attempt to create a NotificationLog row uniquely. If another
    process created it concurrently, fetch and return the existing row.
    Raises IntegrityError only if unexpected.
    """
    log = NotificationLog(
        deadline_id=deadline_id,
        user_id=user_id,
        channel=channel,
        recipient=recipient,
        days_before=days_before,
        status=NotificationStatus.PENDING,
    )
    db.add(log)
    try:
        db.flush()
        db.refresh(log)
        return log
    except IntegrityError:
        db.rollback()
        # Fetch the existing log (could be PENDING or SENT)
        existing = db.query(NotificationLog).filter(
            NotificationLog.deadline_id == deadline_id,
            NotificationLog.days_before == days_before,
            NotificationLog.channel == channel,
        ).first()
        if existing:
            return existing
        # Re-raise if we couldn't find an existing row
        raise


def update_notification_log_by_keys(
    db: Session,
    deadline_id: int,
    days_before: int,
    channel: NotificationChannel,
    status: NotificationStatus,
    message_id: Optional[str] = None,
    error_message: Optional[str] = None,
    message_preview: Optional[str] = None,
) -> Optional[NotificationLog]:
    log = db.query(NotificationLog).filter(
        NotificationLog.deadline_id == deadline_id,
        NotificationLog.days_before == days_before,
        NotificationLog.channel == channel,
    ).first()
    if not log:
        return None
    log.status = status
    if message_id is not None:
        log.message_id = message_id
    if error_message is not None:
        log.error_message = error_message
    if message_preview is not None:
        log.message_preview = message_preview
    if status != NotificationStatus.PENDING:
        log.sent_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    db.refresh(log)
    return log


def create_case_deadline(
    db: Session,
    user_id: int,
    case_id: int,
    case_title: str,
    deadline_date: dt.datetime,
    deadline_type: str,
    description: Optional[str] = None,
) -> CaseDeadline:
    try:
        normalized_case_id = int(case_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("case_id must be an integer matching cases.id") from exc

    # Ownership validation (prevents creating deadlines for other users' cases)
    case = db.query(Case).filter(Case.id == normalized_case_id).first()
    if not case or case.user_id != user_id:
        raise PermissionError(
            "case_id not found or not owned by the provided user_id"
        )
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


def get_upcoming_deadlines(db: Session, days_before: int = 30) -> List[CaseDeadline]:
    now_utc = dt.datetime.now(dt.timezone.utc)
    target_utc = (now_utc + dt.timedelta(days=days_before)).replace(
        hour=23, minute=59, second=59, microsecond=999999
    )

    now = now_utc
    target_date = target_utc
    return db.query(CaseDeadline).filter(
        CaseDeadline.is_completed.is_(False),
        CaseDeadline.deadline_date <= target_date,
        CaseDeadline.deadline_date > now,
    ).all()


def get_prefs_by_user_ids(db: Session, user_ids: Iterable[int]) -> List[UserPreference]:
    user_ids = list(user_ids)
    if not user_ids:
        return []

    return db.query(UserPreference).filter(UserPreference.user_id.in_(user_ids)).all()


def has_notification_been_sent(
    db: Session,
    deadline_id: int,
    days_before: int,
    channel: NotificationChannel,
) -> bool:
    return db.query(NotificationLog).filter(
        NotificationLog.deadline_id == deadline_id,
        NotificationLog.days_before == days_before,
        NotificationLog.channel == channel,
        NotificationLog.status.in_([NotificationStatus.SENT, NotificationStatus.OPENED]),
    ).first() is not None


def log_notification(
    db: Session,
    deadline_id: int,
    user_id: int,
    channel: NotificationChannel,
    recipient: str,
    days_before: int,
    status: NotificationStatus = NotificationStatus.PENDING,
    message_id: Optional[str] = None,
    error_message: Optional[str] = None,
    message_preview: Optional[str] = None,
) -> NotificationLog:
    log = NotificationLog(
        deadline_id=deadline_id,
        user_id=user_id,
        channel=channel,
        recipient=recipient,
        days_before=days_before,
        status=status,
        message_id=message_id,
        error_message=error_message,
        message_preview=message_preview,
        sent_at=dt.datetime.now(dt.timezone.utc) if status != NotificationStatus.PENDING else None,
    )
    db.add(log)
    db.flush()
    db.refresh(log)
    return log


def get_notification_history(db: Session, user_id: int, limit: int = 50) -> List[NotificationLog]:
    return db.query(NotificationLog).filter(
        NotificationLog.user_id == user_id
    ).order_by(NotificationLog.created_at.desc()).limit(limit).all()


def get_notification_template_for_user(db: Session, user_id: int):
    return db.query(NotificationTemplate).filter(NotificationTemplate.user_id == user_id).first()

