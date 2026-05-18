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

2. DISTRIBUTED LOCKING (Redis):
   For horizontally scaled deployments, a Redis-based distributed lock
   ensures exactly-once execution across multiple containers/nodes.
   Only the instance that acquires the lock will execute the job.

3. TIMEZONE-AWARE DISPATCH:
   The system runs an hourly check and calculates the local time for each
   individual user. Reminders are dispatched only when it's 8:00 AM in the
   user's specific timezone (e.g., IST, EST, etc.).

4. FAULT TOLERANCE:
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

DISTRIBUTED LOCKING PATTERN:
--------------------------------------------------------------------------------
- Uses Redis SETNX-based lock with TTL to ensure exclusive job execution
- Lock key format: "legalassist:scheduler:lock"
- Lock TTL: 55 minutes (allows hourly job to complete even if slowest)
- If REDIS_URL env var is not set, falls back to single-instance behavior

================================================================================
"""

import logging
import signal
import sys
import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from contextlib import contextmanager

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
from db import (
    engine,
    init_db,
    SessionLocal,
    get_upcoming_deadlines,
    get_prefs_by_user_ids,
    UserPreference,
)
from notifications.reminder_engine import (
    plan_eligible_reminders,
    should_process_threshold,
    is_notify_enabled,
    is_reminder_time_for_user,
)
from notification_service import NotificationService

# This module is imported by app.py, which handles logging configuration
# Logging setup is centralized in app.py to avoid duplicate handlers

logger = logging.getLogger(__name__)

# Global instances
_scheduler: Optional[BackgroundScheduler] = None
notification_service = NotificationService()
_instance_id = str(uuid.uuid4())[:8]

# Lock configuration
LOCK_KEY = "legalassist:scheduler:lock"
LOCK_TTL_SECONDS = 55 * 60  # 55 minutes to allow hourly job to complete


def _get_redis_client():
    """Get Redis client from REDIS_URL env var."""
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis
        return redis.from_url(redis_url, decode_responses=True)
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        return None


@contextmanager
def distributed_lock(lock_key: str, ttl_seconds: int = 300, lock_id: Optional[str] = None):
    """
    Acquire a distributed lock using Redis SETNX with TTL.

    In horizontally scaled deployments, only the instance that acquires
    the lock will execute the scheduled job, preventing duplicate execution.

    Args:
        lock_key: Redis key for the lock
        ttl_seconds: Time-to-live for the lock (prevents deadlocks if holder crashes)
        lock_id: Unique identifier for this lock holder

    Yields:
        True if lock was acquired, False otherwise
    """
    redis_client = _get_redis_client()
    if redis_client is None:
        yield True
        return

    acquired = False
    if lock_id is None:
        lock_id = f"{_instance_id}:{os.getpid()}"

    try:
        acquired = redis_client.set(lock_key, lock_id, nx=True, ex=ttl_seconds)
        yield acquired
    finally:
        if acquired:
            current_holder = redis_client.get(lock_key)
            if current_holder == lock_id:
                redis_client.delete(lock_key)


def _shutdown_scheduler_instance(scheduler, *, wait: bool = True):
    """Shut down a scheduler instance once, if it is running."""
    if not scheduler:
        return

    try:
        if scheduler.running:
            scheduler.shutdown(wait=wait)
            logger.info("Scheduler shutdown complete.")
    except Exception as e:
        logger.error(f"Error during scheduler shutdown: {e}")


# Reminder time logic moved to notifications.reminder_engine.build_reminder_jobs


def check_and_send_reminders():
    """
    Hourly job: Check all upcoming deadlines and send reminders at 8 AM in each user's local timezone.
    This runs every hour and evaluates if it's 8 AM for each user based on their saved timezone preference.

    Uses distributed locking via Redis to ensure exactly-once execution in horizontally scaled deployments.

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
    1. Distributed Lock Acquisition: Only one instance executes the job
    2. Database Initialization: Ensures tables exist
    3. Data Retrieval: Fetches all deadlines occurring within the next 31 days
    4. Iteration & Filtering:
       a. Computes exact days remaining
       b. Filters to exact thresholds (30, 10, 3, 1)
       c. Fetches user preferences
       d. Evaluates timezone match (is it 8 AM?)
       e. Evaluates preference match (is notify_X_days enabled?)
    5. Dispatch: Hands over to `notification_service` which handles channel-specific logic

    DISTRIBUTED LOCKING:
    -------------------
    In horizontally scaled deployments, the distributed lock ensures that
    only one container/node executes the job. If Redis is not available,
    falls back to single-instance behavior (max_instances=1 still prevents overlap).

    ERROR HANDLING:
    ---------------
    - The entire job is wrapped in a broad try-except block to prevent a single failure
      from crashing the scheduler.
    - Errors are logged with full stack traces.
    - The database session is guaranteed to be closed in the finally block.

    MONITORING:
    -----------
    - The job logs the total number of reminders sent.
    - Lock acquisition status is logged for observability.

    """

    lock_id = f"{_instance_id}:{os.getpid()}"
    with distributed_lock(LOCK_KEY, LOCK_TTL_SECONDS, lock_id) as has_lock:
        if not has_lock:
            logger.debug("Skipping reminder check - another instance holds the lock")
            return

        logger.info("Acquired distributed lock for reminder job")

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

            # Prefetch user preferences for eligible deadlines to avoid N+1 queries
            eligible = []
            for dl in upcoming_deadlines:
                days_left = dl.days_until_deadline()
                if should_process_threshold(days_left):
                    eligible.append((dl, days_left))

            user_ids = {d.user_id for d, _ in eligible}
            prefs_by_user = {}
            if user_ids:
                prefs = db.query(UserPreference).filter(UserPreference.user_id.in_(list(user_ids))).all()
                prefs_by_user = {p.user_id: p for p in prefs}

            for deadline, days_left in eligible:
                user_preference = prefs_by_user.get(deadline.user_id)
                if not user_preference:
                    logger.warning(f"No preferences found for user {deadline.user_id}. Skipping.")
                    continue

                # Check if reminders should be sent based on preferences and time
                if not is_notify_enabled(days_left, user_preference):
                    logger.debug(f"Notifications disabled for this threshold ({days_left} days) for user {deadline.user_id}")
                    continue

                if not is_reminder_time_for_user(user_preference.timezone):
                    logger.debug(
                        f"Not reminder hour yet in user's timezone",
                        user_id=deadline.user_id,
                        user_timezone=user_preference.timezone,
                    )
                    continue

                logger.info(f"Processing deadline: Case={deadline.case_id}, Days Left={days_left}")

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
    _shutdown_scheduler_instance(_scheduler)
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

        _shutdown_scheduler_instance(scheduler)
        sys.exit(0)
    
    # Register SIGINT everywhere Python supports it; SIGTERM is Unix-oriented.
    try:
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers registered.")
    except (ValueError, OSError):
        logger.warning("Could not register signal handlers (not in main thread or unsupported platform).")
    
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
    finally:
        _shutdown_scheduler_instance(scheduler)


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
        prefs = get_prefs_by_user_ids(db, {deadline.user_id for deadline in upcoming_deadlines})
        prefs_by_user = {pref.user_id: pref for pref in prefs}
        candidates = plan_eligible_reminders(
            upcoming_deadlines,
            prefs_by_user,
            reminder_time_checker=is_reminder_time_for_user,
        )
        
        sent_count = 0
        for candidate in candidates:
            deadline = candidate.deadline
            days_left = candidate.days_left

            if target_days and days_left != target_days:
                continue

            # Send reminders
            results = notification_service.send_reminders(db, deadline, candidate.user_preference, days_left)
            sent_count += len([r for r in results if r.success])

        logger.info(f"Synchronous check complete. Reminders sent: {sent_count}")
        return sent_count

    finally:
        if should_close:
            db.close()


if __name__ == "__main__":
    # If run directly, start the worker
    run_worker()
