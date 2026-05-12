"""
================================================================================
LEGALASSIST AI - BACKGROUND JOB SCHEDULING SYSTEM
================================================================================
This module implements the core scheduling logic for the LegalAssist AI 
notification system. It is designed to be resilient, scalable, and 
fault-tolerant by leveraging APScheduler with a persistent database backend.

KEY ARCHITECTURAL COMPONENTS:
--------------------------------------------------------------------------------
1. PERSISTENCE LAYER: 
   Unlike standard in-memory schedulers, this system uses SQLAlchemyJobStore.
   This guarantees that even if the server crashes or the application 
   restarts, the schedule is maintained in the 'apscheduler_jobs' table.

2. TIMEZONE-AWARE DISPATCH:
   The system runs an hourly check and calculates the local time for each 
   individual user. Reminders are dispatched only when it's 8:00 AM in the 
   user's specific timezone (e.g., IST, EST, etc.).

3. FAULT TOLERANCE:
   Uses misfire handling and job coalescing to ensure that if the system 
   goes offline, it catches up on missed notifications without flooding 
   the user with duplicate emails or SMS.

DEPLOYMENT MODES:
--------------------------------------------------------------------------------
- INTEGRATED MODE: The scheduler runs as a background thread within the main
  Streamlit application (via `get_scheduler()`).
- STANDALONE MODE: The scheduler runs as a dedicated worker process 
  (via `run_worker()`), which is the recommended approach for production.

DESIGN PATTERNS USED:
--------------------------------------------------------------------------------
1. Singleton Pattern: `get_scheduler()` ensures only one BackgroundScheduler 
   instance is created when running in integrated mode (e.g., Streamlit).
2. Dependency Injection: The database session (`db`) is instantiated 
   within the job, but the notification service is injected from the global scope.
3. Strategy Pattern: The notification logic delegates to 
   `notification_service` which decides between SMS and Email strategies.

TIMEZONE HANDLING:
--------------------------------------------------------------------------------
- Timezones are a complex domain. We store user preferences as IANA strings
  (e.g., 'America/New_York', 'Asia/Kolkata').
- The `is_reminder_time_for_user` function safely falls back to UTC if a user's
  timezone is invalid or missing.
- By running hourly, we can guarantee that every user will eventually hit 8 AM
  in their local time, exactly once per 24-hour cycle.

================================================================================
"""

import logging
import signal
import sys
import os
from datetime import datetime, timezone
from typing import Optional

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# PERSISTENCE & CONCURRENCY IMPORTS
# ------------------------------------------------------------------------------
# SQLAlchemyJobStore allows us to store job metadata in our primary database.
# ThreadPoolExecutor manages a pool of threads to handle concurrent I/O tasks.
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

# APPLICATION-SPECIFIC IMPORTS
# ------------------------------------------------------------------------------
from database import (
    engine,          # Imported to provide the connection for the job store
    init_db,
    SessionLocal,
    get_upcoming_deadlines,
    UserPreference,
)
from notification_service import NotificationService

# This module is imported by app.py, which handles logging configuration
# Logging setup is centralized in app.py to avoid duplicate handlers

logger = logging.getLogger(__name__)

# Global instances
_scheduler: Optional[BackgroundScheduler] = None
notification_service = NotificationService()


def is_reminder_time_for_user(user_timezone: str, reminder_hour: int = 8) -> bool:

    """

    Check if current time matches the reminder hour in user's local timezone.
    
    Args:
        user_timezone: User's timezone as IANA string (e.g., "Asia/Kolkata")
        reminder_hour: Hour to send reminders (default 8 AM)
    
    Returns:
        True if current time in user's timezone is within the reminder hour
    """
    try:
        if not user_timezone or not isinstance(user_timezone, str):
            raise ValueError("Invalid timezone type")
        tz = pytz.timezone(user_timezone)
        user_now = datetime.now(tz)
        return user_now.hour == reminder_hour
    except (pytz.exceptions.UnknownTimeZoneError, ValueError, AttributeError):
        logger.warning(f"Invalid timezone '{user_timezone}', falling back to UTC")
        # Fallback to UTC if timezone is invalid
        user_now = datetime.now(timezone.utc)
        return user_now.hour == reminder_hour


def check_and_send_reminders():
    """
    Hourly job: Check all upcoming deadlines and send reminders at 8 AM in each user's local timezone.
    This runs every hour and evaluates if it's 8 AM for each user based on their saved timezone preference.
    
    ====================================================================================================
    ARCHITECTURAL OVERVIEW & SCHEDULING STRATEGY
    ====================================================================================================
    
    This function acts as the core heartbeat for the notification system.
    It relies on an hourly execution trigger to ensure that timezone-based
    notifications are dispatched accurately at the start of each user's day (typically 8 AM).
    
    PERFORMANCE OPTIMIZATION:
    -------------------------
    Historically, certain imports (such as `has_notification_been_sent` from `database`) 
    were placed dynamically inside the loop over `upcoming_deadlines`. 
    While localized imports can prevent circular dependencies, placing them inside 
    high-iteration loops introduces significant module resolution overhead.
    
    To alleviate this, we've moved the import to the top of this function.
    This ensures that the `sys.modules` dictionary is only queried once per hourly run,
    rather than O(N) times where N is the number of upcoming deadlines.
    
    PROCESSING WORKFLOW:
    --------------------
    1. Database Initialization: Ensures tables exist.
    2. Data Retrieval: Fetches all deadlines occurring within the next 31 days.
       (We use 31 days to safely capture the 30-day threshold).
    3. Iteration & Filtering:
       a. Computes exact days remaining.
       b. Filters to exact thresholds (30, 10, 3, 1).
       c. Fetches user preferences.
       d. Evaluates timezone match (is it 8 AM?).
       e. Evaluates preference match (is notify_X_days enabled?).
    4. Dispatch: Hands over to `notification_service` which handles channel-specific logic.
    
    SCALABILITY CONSIDERATIONS:
    ---------------------------
    - As the user base grows, fetching all deadlines in memory may become a bottleneck.
    - Future iterations should consider paginating the query or pushing the 
      timezone-filtering logic down to the database level (e.g. using Postgres TIMEZONE functions).
      
    ERROR HANDLING:
    ---------------
    - The entire job is wrapped in a broad try-except block to prevent a single failure
      from crashing the scheduler.
    - Errors are logged with full stack traces.
    - The database session is guaranteed to be closed in the finally block.
    
    MONITORING:
    -----------
    - The job logs the total number of reminders sent.
    - This can be hooked up to Datadog, Prometheus, or simple log alerts to ensure
      the job is actually running and sending emails/SMS.
      
    """
    
    # ---------------------------------------------------------
    # PERFORMANCE FIX: Move localized import out of the loop!
    # ---------------------------------------------------------
    # By placing this import at the top of the function, we avoid
    # the overhead of module resolution during every iteration of
    # the upcoming_deadlines loop. This significantly speeds up
    # the job when processing thousands of deadlines.
    from database import has_notification_been_sent
    # ---------------------------------------------------------

    logger.info("=" * 60)
    logger.info("Starting deadline reminder check job")
    logger.info(f"Check time: {datetime.now(timezone.utc)} UTC")

    # Ensure tables exist when running from a fresh DB.
    init_db()

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
            
            # Check if it's currently 8 AM in the user's local timezone
            if not is_reminder_time_for_user(user_preference.timezone):
                logger.debug(

                    f"Not 8 AM yet in user's timezone",
                    user_id=deadline.user_id,
                    user_timezone=user_preference.timezone,
                )
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
    """
    ============================================================================
    SCHEDULER INITIALIZATION & PERSISTENCE ARCHITECTURE
    ============================================================================
    
    This function is responsible for the bootstrap process of the APScheduler.
    The most significant architectural change here is the move from a volatile,
    RAM-based job store to a durable, database-backed job store using SQLAlchemy.
    
    WHY THIS MIGRATION IS CRITICAL:
    ----------------------------------------------------------------------------
    1. RESILIENCE TO REBOOTS: In containerized environments (like Docker/K8s),
       applications are ephemeral. A restart would wipe out the memory, causing
       the scheduler to "forget" its next run times. With a DB store, the 
       scheduler resumes exactly where it left off.
       
    2. MISFIRE HANDLING: If the application is down when a job was supposed to 
       trigger, the scheduler can detect this "misfire" upon startup by 
       consulting the database. We've configured a 1-hour grace period to 
       ensure these missed tasks are eventually executed.
       
    3. SINGLE-INSTANCE ENFORCEMENT: By setting `max_instances=1`, we ensure 
       that even if a job takes longer than its interval, we don't spawn 
       overlapping tasks, which could lead to duplicate notifications and 
       database race conditions.
    
    PERSISTENCE LAYER CONFIGURATION:
    ----------------------------------------------------------------------------
    We utilize the `apscheduler_jobs` table (automatically managed by 
    APScheduler) within our primary application database. This ensures that 
    the scheduler's state is backed up alongside our application data.
    
    EXECUTION ENGINE:
    ----------------------------------------------------------------------------
    We use a `ThreadPoolExecutor` with a pool size of 20. This is optimized 
    for the I/O-bound nature of our notification system (database queries, 
    SMTP calls, and SMS API requests).
    """
    
    # Log the initialization attempt to the diagnostic logs
    logger.info("Initializing scheduler instance with persistent job store...")

    # 1. DEFINE THE JOB STORE
    # We leverage the existing SQLAlchemy 'engine' from our database module.
    # This avoids managing multiple connection pools and ensures consistent 
    # database configuration across the entire application stack.
    jobstores = {
        'default': SQLAlchemyJobStore(engine=engine)
    }

    # 2. CONFIGURE EXECUTORS
    # A ThreadPoolExecutor is ideal for our workload. We also provide a 
    # ProcessPoolExecutor for CPU-heavy tasks, though it's currently unused.
    executors = {
        'default': ThreadPoolExecutor(20),
        'processpool': ProcessPoolExecutor(5)
    }

    # 3. SET JOB DEFAULTS
    # These settings apply to all jobs unless overridden during add_job.
    job_defaults = {
        'coalesce': True,              # Combine multiple missed runs into one
        'max_instances': 1,            # Prevent overlapping executions of the same job
        'misfire_grace_time': 3600     # 1 hour window to catch up on missed jobs
    }

    # Determine if we are running in background mode (integrated with app)
    # or blocking mode (standalone worker).
    is_background = (scheduler_class == BackgroundScheduler)
    
    # Instantiate the scheduler with our comprehensive configuration
    try:
        scheduler = scheduler_class(
            daemon=is_background,
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=pytz.utc
        )
        
        # 4. REGISTER PERSISTENT JOBS
        # We use a static ID 'deadline_reminder_job' so APScheduler can
        # track this specific job across application restarts in the DB store.
        # replace_existing=True is vital for updating the trigger if we 
        # change the code-level configuration (like the cron schedule).
        scheduler.add_job(
            check_and_send_reminders,
            trigger=CronTrigger(minute=0, second=0),  # Top of every hour
            id="deadline_reminder_job",
            name="Hourly Deadline Reminder Check",
            replace_existing=True
        )
        
        logger.info(f"Successfully configured {scheduler_class.__name__}")
        logger.info("Job store: SQLAlchemy (Persistent)")
        
        return scheduler
        
    except Exception as e:
        logger.critical(f"Failed to initialize scheduler: {str(e)}")
        # If we can't initialize the persistent scheduler, we should probably
        # raise to prevent the application from running in an inconsistent state.
        raise


def get_scheduler():
    """
    Get or create the global background scheduler instance.
    This is the singleton accessor for the scheduler.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = setup_scheduler(BackgroundScheduler)
    return _scheduler


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
    ============================================================================
    STANDALONE WORKER PROCESS
    ============================================================================
    
    This function serves as the entry point for running the scheduler as a 
    dedicated service. In a production environment, this should be managed
    by a process supervisor like systemd, Supervisor, or as a separate
    container in a Kubernetes Pod.
    
    ADVANTAGES OF STANDALONE WORKER:
    ----------------------------------------------------------------------------
    1. ISOLATION: Crashes in the main UI (Streamlit) do not affect the 
       notification engine.
    2. RESOURCE MANAGEMENT: Can be scaled independently of the web frontend.
    3. SIGNAL HANDLING: Properly handles SIGINT and SIGTERM for graceful 
       shutdown, ensuring database connections are closed correctly.
    """
    logger.info("=" * 60)
    logger.info("STARTING LEGALASSIST AI BACKGROUND WORKER")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info("=" * 60)
    
    # Step 1: Ensure the database schema is up to date
    # This is critical if the worker starts before the web app
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Step 2: Initialize the blocking scheduler with persistence
    # BlockingScheduler is used here because this is the main thread of the process
    scheduler = setup_scheduler(BlockingScheduler)
    
    # Step 3: Register signal handlers for graceful termination
    # This ensures that we don't leave zombie jobs or dangling DB connections
    def signal_handler(sig, frame):
        sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {sig_name}. Performing graceful shutdown...")
        
        try:
            # shutdown(wait=True) waits for currently running jobs to finish
            scheduler.shutdown(wait=True)
            logger.info("Scheduler shutdown complete.")
        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {e}")
            
        sys.exit(0)
    
    # Registering signals (note: limited support on Windows)
    if os.name != 'nt':
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.info("UNIX signal handlers registered.")
        except ValueError:
            logger.warning("Could not register signal handlers (not in main thread).")
    else:
        logger.info("Running on Windows: Use Ctrl+C for manual termination.")
    
    logger.info("Worker initialization complete. Entering wait loop.")
    logger.info("Next job run scheduled at the start of the next hour.")
    
    try:
        # This will block until the process is terminated
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker stopped by user or system.")
    except Exception as e:
        logger.critical(f"Worker encountered a fatal error: {e}", exc_info=True)
        sys.exit(1)


def check_reminders_sync(target_days: Optional[int] = None, db: Optional[object] = None):
    """
    Synchronous version for testing. Optionally check only specific day threshold.
    Args:
        target_days: If specified, only check this day threshold (e.g., 30, 10, 3, 1)
        db: Optional database session. If not provided, uses SessionLocal()
    """
    should_close = False
    if db is None:
        db = SessionLocal()
        should_close = True
    
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
        if should_close:
            db.close()


if __name__ == "__main__":
    # If run directly, start the worker
    run_worker()
