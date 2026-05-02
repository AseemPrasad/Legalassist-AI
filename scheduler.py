"""
Background job scheduler for sending deadline reminders.
Uses APScheduler to run daily checks for upcoming deadlines.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
import pytz

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from database import (
    SessionLocal,
    get_upcoming_deadlines,
    NotificationChannel,
    CaseDeadline,
    UserPreference,
    has_notification_been_sent,
)
from notification_service import NotificationService

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler: Optional[BackgroundScheduler] = None
notification_service = NotificationService()


def check_and_send_reminders():
    """
    Daily job: Check all upcoming deadlines and send reminders.
    This runs at 8 AM UTC and checks for deadlines at 30, 10, 3, and 1 day marks.
    """
    logger.info("=" * 60)
    logger.info("Starting deadline reminder check job")
    logger.info(f"Check time: {datetime.now(timezone.utc)}")

    db = SessionLocal()
    try:
        # Import here to avoid circular imports
        from database import has_notification_been_sent
        
        # Check for deadlines in the next 31 days to ensure we catch the 30-day mark
        upcoming_deadlines = get_upcoming_deadlines(db, days_before=31)
        logger.info(f"Found {len(upcoming_deadlines)} upcoming deadlines")

        for deadline in upcoming_deadlines:
            days_left = deadline.days_until_deadline()
            
            # Only process deadlines at reminder thresholds
            if days_left not in [30, 10, 3, 1]:
                continue

            logger.info(f"Processing deadline: Case={deadline.case_id}, Days Left={days_left}")

            # Get user preferences
            user_preference = db.query(UserPreference).filter(
                UserPreference.user_id == deadline.user_id
            ).first()

            if not user_preference:
                logger.warning(f"No preferences found for user {deadline.user_id}. Skipping.")
                continue

            # Check if reminders should be sent based on preferences
            should_notify = False
            if days_left == 30 and user_preference.notify_30_days:
                should_notify = True
            if days_left == 10 and user_preference.notify_10_days:
                should_notify = True
            if days_left == 3 and user_preference.notify_3_days:
                should_notify = True
            if days_left == 1 and user_preference.notify_1_day:
                should_notify = True

            if not should_notify:
                logger.debug(f"Notifications disabled for this threshold ({days_left} days)")
                continue

            # Check if reminder was already sent
            if user_preference.notification_channel in [NotificationChannel.SMS, NotificationChannel.BOTH]:
                if has_notification_been_sent(db, deadline.id, days_left, NotificationChannel.SMS):
                    logger.debug(f"SMS reminder already sent for deadline {deadline.id}")
                else:
                    result = notification_service.send_sms_reminder(db, deadline, user_preference, days_left)
                    status = "✓" if result.success else "✗"
                    logger.info(f"{status} SMS sent to {result.recipient}")

            if user_preference.notification_channel in [NotificationChannel.EMAIL, NotificationChannel.BOTH]:
                if has_notification_been_sent(db, deadline.id, days_left, NotificationChannel.EMAIL):
                    logger.debug(f"Email reminder already sent for deadline {deadline.id}")
                else:
                    result = notification_service.send_email_reminder(db, deadline, user_preference, days_left)
                    status = "✓" if result.success else "✗"
                    logger.info(f"{status} Email sent to {result.recipient}")

        logger.info("Deadline reminder check job completed successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in reminder job: {str(e)}", exc_info=True)
    finally:
        db.close()


def get_scheduler() -> BackgroundScheduler:
    """Get or create the background scheduler"""
    global _scheduler
    
    if _scheduler is None:
        _scheduler = BackgroundScheduler(daemon=True)
        
        # Schedule daily job at 8 AM UTC
        # You can customize the timezone by setting pytz timezone
        _scheduler.add_job(
            check_and_send_reminders,
            trigger=CronTrigger(hour=8, minute=0, second=0),  # 8 AM UTC daily
            id="deadline_reminder_job",
            name="Daily Deadline Reminder Check",
            replace_existing=True,
            misfire_grace_time=300,  # 5 minute grace for misfires
        )
        
        logger.info("Scheduler initialized. Job scheduled for 8:00 AM UTC daily.")
    
    return _scheduler


def start_scheduler():
    """Start the background scheduler"""
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
        logger.info("Background scheduler started")
    else:
        logger.info("Scheduler already running")


def stop_scheduler():
    """Stop the background scheduler"""
    global _scheduler
    
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        _scheduler = None
        logger.info("Background scheduler stopped")


def trigger_reminder_check_now():
    """
    Manually trigger the reminder check (useful for testing/debugging).
    Will be run directly without waiting for scheduled time.
    """
    logger.info("Manually triggering reminder check...")
    check_and_send_reminders()


# For development/testing: synchronous version
def check_reminders_sync(target_days: Optional[int] = None):
    """
    Synchronous version for testing. Optionally check only specific day threshold.
    Args:
        target_days: If specified, only check this day threshold (e.g., 30, 10, 3, 1)
    """
    from database import has_notification_been_sent
    
    db = SessionLocal()
    try:
        logger.info(f"Running synchronous reminder check (target_days={target_days})")
        upcoming_deadlines = get_upcoming_deadlines(db, days_before=31)
        
        sent_count = 0
        for deadline in upcoming_deadlines:
            days_left = deadline.days_until_deadline()
            
            if target_days and days_left != target_days:
                continue
            
            if days_left not in [30, 10, 3, 1]:
                continue

            user_preference = db.query(UserPreference).filter(
                UserPreference.user_id == deadline.user_id
            ).first()

            if not user_preference:
                continue

            # Send reminders
            results = notification_service.send_reminders(db, deadline, user_preference)
            sent_count += len([r for r in results if r.success])

        logger.info(f"Synchronous check complete. Reminders sent: {sent_count}")
        return sent_count

    finally:
        db.close()
