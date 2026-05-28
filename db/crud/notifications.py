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
) -> tuple[NotificationLog, bool]:
    """Atomically create a NotificationLog row under a savepoint.

    Uses a nested transaction (savepoint) so the unique constraint on
    (deadline_id, days_before, channel) is enforced immediately via flush,
    and IntegrityError is caught within the function itself.  Without a
    savepoint, two concurrent readers can both flush() the same key under
    READ COMMITTED isolation and both observe a successful insert; the
    IntegrityError would only surface at the outer commit, outside this
    function's exception handler.
    """
    try:
        with db.begin_nested():
            log = NotificationLog(
                deadline_id=deadline_id,
                user_id=user_id,
                channel=channel,
                recipient=recipient,
                days_before=days_before,
                status=NotificationStatus.PENDING,
            )
            db.add(log)
            db.flush()
        db.refresh(log)
        return log, True
    except IntegrityError:
        existing = db.query(NotificationLog).filter(
            NotificationLog.deadline_id == deadline_id,
            NotificationLog.days_before == days_before,
            NotificationLog.channel == channel,
        ).first()
        if existing:
            return existing, False
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
    now = dt.datetime.now(dt.timezone.utc)
    if status == NotificationStatus.SENT:
        log.sent_at = now
    elif status == NotificationStatus.DELIVERED:
        log.delivered_at = now
    elif status == NotificationStatus.FAILED:
        log.failed_at = now
    elif status != NotificationStatus.PENDING:
        log.sent_at = now
    db.commit()
    db.refresh(log)
    return log


def update_notification_log_by_message_id(
    db: Session,
    message_id: str,
    status: NotificationStatus,
    error_message: Optional[str] = None,
    message_preview: Optional[str] = None,
) -> Optional[NotificationLog]:
    log = db.query(NotificationLog).filter(NotificationLog.message_id == message_id).first()
    if not log:
        return None

    log.status = status
    if error_message is not None:
        log.error_message = error_message
    if message_preview is not None:
        log.message_preview = message_preview

    now = dt.datetime.now(dt.timezone.utc)
    if status == NotificationStatus.DELIVERED:
        log.delivered_at = now
    elif status == NotificationStatus.FAILED:
        log.failed_at = now
    elif status == NotificationStatus.SENT:
        log.sent_at = now

    db.commit()
    db.refresh(log)
    return log


def reserve_notification(
    db: Session,
    deadline_id: int,
    user_id: int,
    channel: NotificationChannel,
    recipient: str,
    days_before: int,
    message_preview: Optional[str] = None,
) -> tuple[NotificationLog, bool]:
    """Reserve a pending notification row before delivery starts."""
    log = NotificationLog(
        deadline_id=deadline_id,
        user_id=user_id,
        channel=channel,
        recipient=recipient,
        days_before=days_before,
        status=NotificationStatus.PENDING,
        message_preview=message_preview,
    )
    try:
        db.add(log)
        db.commit()
        db.refresh(log)
        return log, True
    except IntegrityError:
        db.rollback()
        existing = db.query(NotificationLog).filter(
            NotificationLog.deadline_id == deadline_id,
            NotificationLog.days_before == days_before,
            NotificationLog.channel == channel,
        ).first()
        return existing, False


def update_notification_result(
    db: Session,
    deadline_id: int,
    user_id: int,
    days_before: int,
    channel: NotificationChannel,
    status: NotificationStatus,
    message_id: Optional[str] = None,
    error_message: Optional[str] = None,
    message_preview: Optional[str] = None,
) -> NotificationLog:
    """Upsert a notification log after a delivery attempt."""
    existing = db.query(NotificationLog).filter(
        NotificationLog.deadline_id == deadline_id,
        NotificationLog.days_before == days_before,
        NotificationLog.channel == channel,
    ).with_for_update(read=True).first()

    if existing:
        existing.status = status
        existing.message_id = message_id or existing.message_id
        existing.error_message = error_message or existing.error_message
        existing.message_preview = message_preview or existing.message_preview
        if status == NotificationStatus.SENT:
            existing.sent_at = dt.datetime.now(dt.timezone.utc)
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing

    return get_or_create_notification_log(
        db=db,
        deadline_id=deadline_id,
        user_id=user_id,
        channel=channel,
        recipient="unknown",
        days_before=days_before,
    )[0]


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
        NotificationLog.status.in_([
            NotificationStatus.SENT,
            NotificationStatus.DELIVERED,
            NotificationStatus.OPENED,
        ]),
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
        sent_at=dt.datetime.now(dt.timezone.utc) if status == NotificationStatus.SENT else None,
        delivered_at=dt.datetime.now(dt.timezone.utc) if status == NotificationStatus.DELIVERED else None,
        failed_at=dt.datetime.now(dt.timezone.utc) if status == NotificationStatus.FAILED else None,
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

