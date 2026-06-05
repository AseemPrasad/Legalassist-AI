from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy.orm import Session

from db.models import CaseDeadline, NotificationChannel, NotificationTemplate, UserPreference


def create_or_update_user_preference(
    db: Session,
    user_id: int,
    email: str,
    phone_number: Optional[str] = None,
    notification_channel: NotificationChannel = NotificationChannel.BOTH,
    timezone: str = "UTC",
    holiday_aware_reminders: bool = False,
    holiday_country: Optional[str] = None,
    holiday_region: Optional[str] = None,
    holiday_calendar_json: Optional[str] = None,
    reminder_thresholds: Optional[list[int]] = None,
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
        if reminder_thresholds is not None:
            pref.reminder_thresholds = reminder_thresholds
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
            reminder_thresholds=reminder_thresholds,
        )
        db.add(pref)

    db.commit()
    db.refresh(pref)
    return pref


def get_notification_template_for_user(db: Session, user_id: int) -> Optional[NotificationTemplate]:
    """Get notification template for a user"""
    return db.query(NotificationTemplate).filter(NotificationTemplate.user_id == user_id).first()


def get_user_deadlines(db: Session, user_id: int):
    """Get all active deadlines for a user"""
    return db.query(CaseDeadline).filter(
        CaseDeadline.user_id == user_id,
        CaseDeadline.status == "active",
    ).order_by(CaseDeadline.deadline_date).all()


def check_and_update_overdue_deadlines(db: Session) -> int:
    """
    Find all active deadlines that have passed their deadline_date and transition them to overdue.
    Returns the number of transitioned deadlines.
    """
    import logging
    from db.models.cases import CaseDeadline
    from db.case_service import transition_deadline

    logger = logging.getLogger(__name__)
    now = dt.datetime.now(dt.timezone.utc)
    
    # Query active deadlines that are past due
    overdue_deadlines = db.query(CaseDeadline).filter(
        CaseDeadline.status == "active",
        CaseDeadline.deadline_date <= now
    ).all()

    count = 0
    for deadline in overdue_deadlines:
        try:
            # Transition status to overdue (actor is system, we use user_id of the deadline owner)
            transition_deadline(db, deadline.id, "overdue", actor_user_id=deadline.user_id)
            count += 1
        except Exception as e:
            logger.error(
                "failed_to_auto_transition_overdue_deadline",
                deadline_id=deadline.id,
                error=str(e),
                exc_info=True
            )
    return count
