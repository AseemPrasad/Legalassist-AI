"""
Background job scheduler for sending deadline reminders.
Uses APScheduler to run daily checks for upcoming deadlines.
Can be run as a standalone worker or integrated into an application.
"""

import logging
import signal
import sys
import os
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from database import (
    init_db,
    SessionLocal,
    get_upcoming_deadlines,
    UserPreference,
)
from notification_service import NotificationService

# Configure logging for standalone mode
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global instances
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
        # Check for deadlines in the next 31 days to ensure we catch the 30-day mark
        upcoming_deadlines = get_upcoming_deadlines(db, days_before=31)
        logger.info(f"Found {len(upcoming_deadlines)} upcoming deadlines")

        sent_count = 0
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

            # Send reminders using the notification service
            results = notification_service.send_reminders(db, deadline, user_preference, days_left)
            
            for res in results:
                if res.success:
                    sent_count += 1
                    logger.info(f"✓ {res.channel.upper()} sent to {res.recipient}")
                else:
                    logger.error(f"✗ {res.channel.upper()} failed for {res.recipient}: {res.error}")

        logger.info(f"Deadline reminder check job completed. Total reminders sent: {sent_count}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in reminder job: {str(e)}", exc_info=True)
    finally:
        db.close()


def setup_scheduler(scheduler_class):
    """Initialize and configure a scheduler instance"""
    # BackgroundScheduler needs daemon=True, BlockingScheduler does not
    is_background = (scheduler_class == BackgroundScheduler)
    scheduler = scheduler_class(daemon=is_background)
    
    # Schedule daily job at 8 AM UTC
    scheduler.add_job(
        check_and_send_reminders,
        trigger=CronTrigger(hour=8, minute=0, second=0),  # 8 AM UTC daily
        id="deadline_reminder_job",
        name="Daily Deadline Reminder Check",
        replace_existing=True,
        misfire_grace_time=300,  # 5 minute grace for misfires
    )
    
    return scheduler


def start_scheduler():
    """
    Start the background scheduler (legacy support for app.py).
    Note: Moving to standalone worker is recommended for production.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = setup_scheduler(BackgroundScheduler)
    
    if not _scheduler.running:
        _scheduler.start()
        logger.info("Background scheduler started (integrated mode)")
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
    """
    logger.info("Manually triggering reminder check...")
    check_and_send_reminders()


def run_worker():
    """
    Run the scheduler as a standalone blocking worker process.
    This is the preferred way to run background tasks in production.
    """
    logger.info("Starting LegalAssist AI Worker...")
    
    # Initialize database
    init_db()
    
    # Setup blocking scheduler
    scheduler = setup_scheduler(BlockingScheduler)
    
    # Signal handling for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down worker...")
        scheduler.shutdown()
        sys.exit(0)
    
    # Only register signals if we are in the main thread (standalone mode)
    if os.name != 'nt': # Signals have limited support on Windows
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            logger.warning("Could not register signal handlers (not in main thread?)")
    
    logger.info("Worker initialized. Job scheduled for 08:00 UTC daily.")
    logger.info("Press Ctrl+C to exit.")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker stopped.")


def check_reminders_sync(target_days: Optional[int] = None):
    """
    Synchronous version for testing. Optionally check only specific day threshold.
    Args:
        target_days: If specified, only check this day threshold (e.g., 30, 10, 3, 1)
    """
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
            results = notification_service.send_reminders(db, deadline, user_preference, days_left)
            sent_count += len([r for r in results if r.success])

        logger.info(f"Synchronous check complete. Reminders sent: {sent_count}")
        return sent_count

    finally:
        db.close()


if __name__ == "__main__":
    # If run directly, start the worker
    run_worker()
