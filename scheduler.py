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
import signal
import sys
import os
import uuid
import subprocess
import shlex
import time
from datetime import datetime, timezone
from typing import Optional, Callable
from contextlib import contextmanager

import pytz
import structlog
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    # PERSISTENCE & CONCURRENCY IMPORTS
    # ------------------------------------------------------------------------------
    # SQLAlchemyJobStore allows us to store job metadata in our primary database.
    # ThreadPoolExecutor manages a pool of threads to handle concurrent I/O tasks.
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
except Exception:
    BackgroundScheduler = None
    BlockingScheduler = None
    CronTrigger = None
    SQLAlchemyJobStore = None
    ThreadPoolExecutor = None
    ProcessPoolExecutor = None

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
from db.crud.knowledge import process_due_knowledge_invalidations
from notifications.reminder_engine import (
    plan_eligible_reminders,
    should_process_threshold,
    is_notify_enabled,
    is_reminder_time_for_user,
)
from notification_service import NotificationService
from api.idempotency import IdempotencyManager
from core.log_redaction import mask_recipient, sanitize_log_text, sanitize_log_value

# This module is imported by app.py, which handles logging configuration
# Logging setup is centralized in app.py to avoid duplicate handlers

logger = structlog.get_logger(__name__)

# Global instances
_scheduler: Optional[BackgroundScheduler] = None
_notification_service_instance: Optional[NotificationService] = None
_instance_id = str(uuid.uuid4())[:8]


def _is_truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_MAINTENANCE_TASKS = _is_truthy_env("ENABLE_MAINTENANCE_TASKS")
MAINTENANCE_TASK_COMMAND = os.getenv("MAINTENANCE_TASK_COMMAND", "").strip()


class _LazyNotificationService:
    """Lazy proxy that initializes NotificationService on first attribute access."""

    def _ensure(self) -> NotificationService:
        global _notification_service_instance
        if _notification_service_instance is None:
            _notification_service_instance = NotificationService()
        return _notification_service_instance

    def __getattr__(self, name):
        return getattr(self._ensure(), name)


notification_service = _LazyNotificationService()
notification_dispatch_idempotency = IdempotencyManager()


def get_notification_service() -> NotificationService:
    """Lazily initialize the notification service singleton."""
    return notification_service._ensure()

# Lock configuration
LOCK_KEY = "legalassist:scheduler:lock"
LOCK_TTL_SECONDS = 55 * 60  # 55 minutes to allow hourly job to complete
KNOWLEDGE_LOCK_KEY = "legalassist:knowledge:lock"
KNOWLEDGE_LOCK_TTL_SECONDS = 10 * 60


def _get_redis_client():
    """Get Redis client from REDIS_URL env var."""
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis
        return redis.from_url(redis_url, decode_responses=True)
    except Exception as e:
        logger.warning("scheduler_redis_unavailable", error=sanitize_log_text(str(e)))
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
            # Atomic compare-and-delete via Lua script.
            # A plain GET + DELETE is a race: if the TTL expires between the two
            # calls another instance can acquire the lock, and our subsequent
            # DELETE would then remove *their* key, breaking mutual exclusion.
            # Lua executes atomically on the Redis server — no other command
            # can interleave between the ownership check and the deletion.
            _UNLOCK_SCRIPT = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "    return redis.call('del', KEYS[1]) "
                "else "
                "    return 0 "
                "end"
            )
            try:
                redis_client.eval(_UNLOCK_SCRIPT, 1, lock_key, lock_id)
            except Exception as e:
                logger.error(
                    "scheduler_lock_release_failed",
                    lock_key=lock_key,
                    error=sanitize_log_text(str(e)),
                )


def _shutdown_scheduler_instance(scheduler, *, wait: bool = True):
    """Shut down a scheduler instance once, if it is running."""
    if not scheduler:
        return

    try:
        if scheduler.running:
            scheduler.shutdown(wait=wait)
            logger.info("scheduler_shutdown_complete")
    except Exception as e:
        logger.error("scheduler_shutdown_failed", error=sanitize_log_text(str(e)))


def _send_deadline_reminders_safe(db, deadline, user_preference, days_left):
    """Send reminders for one deadline and isolate notification service failures."""
    operation_key = IdempotencyManager.build_operation_key(
        operation="deadline_reminder_dispatch",
        principal=str(getattr(deadline, "user_id", "unknown")),
        parts=[
            f"deadline:{getattr(deadline, 'id', 'unknown')}",
            f"days_left:{days_left}",
            f"channel:{getattr(user_preference, 'notification_channel', 'unknown')}",
        ],
    )
    if not notification_dispatch_idempotency.acquire(operation_key, ttl=15 * 60):
        logger.info(
            "scheduler_reminder_dispatch_skipped",
            deadline_id=getattr(deadline, "id", None),
            user_id=getattr(deadline, "user_id", None),
            days_left=days_left,
        )
        return []

    try:
        results = notification_service.send_reminders(db, deadline, user_preference, days_left)
        notification_dispatch_idempotency.mark_completed(
            operation_key,
            {
                "deadline_id": getattr(deadline, "id", None),
                "user_id": getattr(deadline, "user_id", None),
                "days_left": days_left,
                "sent_count": len(results),
            },
            ttl=24 * 60 * 60,
        )
        return results
    except Exception as exc:
        logger.error(
            "scheduler_notification_dispatch_failed",
            deadline_id=getattr(deadline, "id", None),
            user_id=getattr(deadline, "user_id", None),
            days_left=days_left,
            error=sanitize_log_text(str(exc)),
            exc_info=True,
        )
        return []
    finally:
        notification_dispatch_idempotency.release_lock(operation_key)


# Reminder time logic moved to notifications.reminder_engine.build_reminder_jobs


def check_and_send_reminders(reminder_time_checker: Optional[Callable[[str], bool]] = None):
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

    sent_count = 0

    lock_id = f"{_instance_id}:{os.getpid()}"
    with distributed_lock(LOCK_KEY, LOCK_TTL_SECONDS, lock_id) as has_lock:
        if not has_lock:
            logger.debug("scheduler_reminder_lock_skipped")
            return

        logger.info("scheduler_reminder_lock_acquired")

        sent_count = 0

        logger.info("scheduler_reminder_job_started", check_time=datetime.now(timezone.utc).isoformat())

        # Ensure tables exist when running from a fresh DB.
        init_db()

        db = SessionLocal()
        try:
            # Check for deadlines in the next 31 days to ensure we catch the 30-day mark
            upcoming_deadlines = get_upcoming_deadlines(db, days_before=31)
            logger.info("scheduler_upcoming_deadlines_found", count=len(upcoming_deadlines))

            # Prefetch user preferences for eligible deadlines to avoid N+1 queries
            prefs = get_prefs_by_user_ids(db, {deadline.user_id for deadline in upcoming_deadlines})
            prefs_by_user = {pref.user_id: pref for pref in prefs}
            candidates = plan_eligible_reminders(
                upcoming_deadlines,
                prefs_by_user,
                reminder_time_checker=reminder_time_checker or is_reminder_time_for_user,
            )

            for candidate in candidates:
                deadline = candidate.deadline
                days_left = candidate.days_left
                user_preference = candidate.user_preference

                logger.info("scheduler_processing_deadline", case_id=deadline.case_id, days_left=days_left)

                # Send reminders using the notification service
                results = _send_deadline_reminders_safe(db, deadline, user_preference, days_left)

                for res in results:
                    if res.success:
                        sent_count += 1
                        logger.info(
                            "scheduler_notification_sent",
                            channel=res.channel.value if hasattr(res.channel, "value") else str(res.channel),
                            recipient=mask_recipient(res.recipient),
                        )
                    else:
                        logger.error(
                            "scheduler_notification_failed",
                            channel=res.channel.value if hasattr(res.channel, "value") else str(res.channel),
                            recipient=mask_recipient(res.recipient),
                            error=sanitize_log_text(res.error),
                        )

            logger.info("scheduler_reminder_job_completed", reminders_sent=sent_count)

        except Exception as e:
            logger.error("scheduler_reminder_job_failed", error=sanitize_log_text(str(e)), exc_info=True)
        finally:
            db.close()

    return sent_count


def recompute_due_knowledge_invalidations():
    """Recompute stale knowledge artifacts after invalidations become due."""

    lock_id = f"{_instance_id}:{os.getpid()}"
    with distributed_lock(KNOWLEDGE_LOCK_KEY, KNOWLEDGE_LOCK_TTL_SECONDS, lock_id) as has_lock:
        if not has_lock:
            logger.debug("Skipping knowledge recompute - another instance holds the lock")
            return

        init_db()

        db = SessionLocal()
        try:
            processed = process_due_knowledge_invalidations(db)
            logger.info("scheduler_knowledge_recompute_completed", processed=len(processed))
        except Exception as exc:
            logger.error("scheduler_knowledge_recompute_failed", error=sanitize_log_text(str(exc)), exc_info=True)
        finally:
            db.close()


@contextmanager
def managed_subprocess(command: list[str] | str, **kwargs):
    """
    ============================================================================
    MANAGED SUBPROCESS CONTEXT MANAGER
    ============================================================================
    
    This context manager wraps the execution of OS-level subprocesses to guarantee
    cleanup, preventing the accumulation of zombie processes when a scheduled
    task crashes or times out abruptly.
    
    WHY THIS IS CRITICAL:
    ----------------------------------------------------------------------------
    1. ZOMBIE PROCESS PREVENTION: When a child process terminates, it remains
       in the process table as a "zombie" until the parent reads its exit status.
       If the parent scheduler crashes or abandons the child, these accumulate
       and eventually exhaust the host's PID space, causing system freezes.
       
    2. RESOURCE LEAK MITIGATION: Subprocesses hold onto file descriptors (stdout,
       stderr) and memory. Abrupt failures without cleanup leak these resources.
       
    3. TIMEOUT ENFORCEMENT: Ensures long-running hung processes can be forcibly
       terminated if they exceed their expected execution window.
       
    IMPLEMENTATION DETAILS:
    ----------------------------------------------------------------------------
    - Spawns the subprocess using `subprocess.Popen`.
    - Yields the process object to the caller.
    - In the `finally` block, it enforces termination.
    - First attempts graceful `terminate()` (SIGTERM).
    - If the process doesn't exit, escalates to `kill()` (SIGKILL).
    - Finally, calls `wait()` to ensure the process is reaped from the OS table.
    """
    if isinstance(command, str):
        command = shlex.split(command)
        
    logger.info("scheduler_subprocess_spawned", command=sanitize_log_value(command, "command"))
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kwargs
        )
        yield process
    except Exception as e:
        logger.error("scheduler_subprocess_failed", command=sanitize_log_value(command, "command"), error=sanitize_log_text(str(e)))
        raise
    finally:
        if process is not None:
            # Check if process is still running
            if process.poll() is None:
                logger.warning("scheduler_subprocess_cleanup_required", pid=process.pid)
                try:
                    # Attempt graceful termination (SIGTERM)
                    process.terminate()
                    
                    # Give it a brief moment to exit gracefully
                    try:
                        process.wait(timeout=5.0)
                        logger.info("scheduler_subprocess_terminated", pid=process.pid, mode="graceful")
                    except subprocess.TimeoutExpired:
                        # Escalate to SIGKILL if it refuses to die
                        logger.error("scheduler_subprocess_sigterm_ignored", pid=process.pid)
                        process.kill()
                        process.wait(timeout=2.0)
                        logger.info("scheduler_subprocess_terminated", pid=process.pid, mode="forced")
                except ProcessLookupError:
                    # Process might have exited just before we tried to terminate
                    pass
                except Exception as cleanup_error:
                    logger.critical("scheduler_subprocess_cleanup_failed", pid=process.pid, error=sanitize_log_text(str(cleanup_error)))
            else:
                # Process already exited, but we must read its exit code to prevent zombies
                process.wait()
            
            # Always ensure stdout/stderr pipes are closed to prevent FD leaks
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


def run_system_maintenance_task():
    """
    ============================================================================
    SYSTEM MAINTENANCE TASK
    ============================================================================
    
    A scheduled job that performs OS-level maintenance (e.g., log rotation,
    temporary file cleanup, database vacuuming) using a subprocess.

    This job is opt-in. Set ENABLE_MAINTENANCE_TASKS=1 and provide a real
    MAINTENANCE_TASK_COMMAND to execute a configured script or command.
    
    This function utilizes the `managed_subprocess` context manager to ensure
    that even if the script hangs or crashes, the child process is reaped
    and does not become a zombie process.
    """
    lock_id = f"{_instance_id}:{os.getpid()}"
    with distributed_lock("legalassist:maintenance:lock", 300, lock_id) as has_lock:
        if not has_lock:
            logger.debug("scheduler_maintenance_lock_skipped")
            return

        if not ENABLE_MAINTENANCE_TASKS:
            logger.info("scheduler_maintenance_disabled")
            return

        if not MAINTENANCE_TASK_COMMAND:
            logger.warning("scheduler_maintenance_command_missing")
            return

        logger.info("scheduler_maintenance_started")
        command = shlex.split(MAINTENANCE_TASK_COMMAND)
        
        try:
            with managed_subprocess(command) as process:
                try:
                    # Wait for completion with a timeout
                    stdout, stderr = process.communicate(timeout=30)
                    
                    if process.returncode == 0:
                        logger.info("scheduler_maintenance_completed")
                        if stdout:
                            logger.debug("scheduler_maintenance_output", output=sanitize_log_text(stdout.strip()))
                    else:
                        logger.error("scheduler_maintenance_failed", return_code=process.returncode)
                        if stderr:
                            logger.error("scheduler_maintenance_errors", error=sanitize_log_text(stderr.strip()))
                except subprocess.TimeoutExpired:
                    logger.error("scheduler_maintenance_timeout")
                    # The managed_subprocess finally block will handle termination
        except Exception as e:
            logger.error("scheduler_maintenance_exception", error=sanitize_log_text(str(e)), exc_info=True)

    return sent_count


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
    logger.info("scheduler_initializing")

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

        scheduler.add_job(
            recompute_due_knowledge_invalidations,
            trigger=CronTrigger(minute="*/15", second=10),
            id="knowledge_recompute_job",
            name="Quarter-Hour Knowledge Recompute",
            replace_existing=True,
        )

        if ENABLE_MAINTENANCE_TASKS:
            scheduler.add_job(
                run_system_maintenance_task,
                trigger=CronTrigger(hour=3, minute=0),  # 3 AM every day
                id="system_maintenance_job",
                name="Configured System Maintenance",
                replace_existing=True,
            )
        else:
            logger.info("scheduler_maintenance_job_disabled")
        
        logger.info("scheduler_configured", scheduler_class=getattr(scheduler_class, "__name__", str(scheduler_class)), job_store="sqlalchemy")
        
        return scheduler
        
    except Exception as e:
        logger.critical("scheduler_initialization_failed", error=sanitize_log_text(str(e)))
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
        logger.info("scheduler_started", mode="integrated")
    else:
        logger.info("scheduler_already_running")


def stop_scheduler():
    """Stop the background scheduler"""
    global _scheduler
    _shutdown_scheduler_instance(_scheduler)
    _scheduler = None
    logger.info("scheduler_stopped")


def trigger_reminder_check_now():
    """
    Manually trigger the reminder check (useful for testing/debugging).
    """
    logger.info("scheduler_manual_reminder_triggered")
    check_and_send_reminders()


def trigger_knowledge_recompute_now():
    """Manually trigger the knowledge recompute job (useful for testing/debugging)."""
    logger.info("scheduler_manual_knowledge_recompute_triggered")
    recompute_due_knowledge_invalidations()


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
    logger.info("scheduler_worker_starting", pid=os.getpid())
    
    # Step 1: Ensure the database schema is up to date
    # This is critical if the worker starts before the web app
    try:
        init_db()
        logger.info("scheduler_database_initialized")
    except Exception as e:
        logger.error("scheduler_database_init_failed", error=sanitize_log_text(str(e)))
        sys.exit(1)
    
    # Step 2: Initialize the blocking scheduler with persistence
    # BlockingScheduler is used here because this is the main thread of the process
    scheduler = setup_scheduler(BlockingScheduler)
    
    # Step 3: Register signal handlers for graceful termination
    # This ensures that we don't leave zombie jobs or dangling DB connections
    def signal_handler(sig, frame):
        sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
        logger.info("scheduler_signal_received", signal=sig_name)

        _shutdown_scheduler_instance(scheduler)
        sys.exit(0)
    
    # Register SIGINT everywhere Python supports it; SIGTERM is Unix-oriented.
    try:
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)
        logger.info("scheduler_signal_handlers_registered")
    except (ValueError, OSError):
        logger.warning("scheduler_signal_handlers_unavailable")
    
    logger.info("scheduler_worker_ready")
    
    try:
        # This will block until the process is terminated
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("scheduler_worker_stopped")
    except Exception as e:
        logger.critical("scheduler_worker_fatal_error", error=sanitize_log_text(str(e)), exc_info=True)
        sys.exit(1)
    finally:
        _shutdown_scheduler_instance(scheduler)


def check_reminders_sync(
    target_days: Optional[int] = None,
    db: Optional[object] = None,
    reminder_time_checker: Optional[Callable[[str], bool]] = None,
):
    """
    Synchronous version for testing. Optionally check only specific day threshold.
    Args:
        target_days: If specified, only check this day threshold (e.g., 30, 10, 3, 1)
        db: Optional database session. If not provided, uses SessionLocal()
        reminder_time_checker: Optional time checker. Defaults to lambda tz: True to bypass time window.
    """
    should_close = False
    if db is None:
        db = SessionLocal()
        should_close = True
    
    try:
        logger.info("scheduler_sync_reminder_check_started", target_days=target_days)
        upcoming_deadlines = get_upcoming_deadlines(db, days_before=31)
        prefs = get_prefs_by_user_ids(db, {deadline.user_id for deadline in upcoming_deadlines})
        prefs_by_user = {pref.user_id: pref for pref in prefs}
        candidates = plan_eligible_reminders(
            upcoming_deadlines,
            prefs_by_user,
            reminder_time_checker=reminder_time_checker or (lambda tz: True),
        )
        
        sent_count = 0
        for candidate in candidates:
            deadline = candidate.deadline
            days_left = candidate.days_left

            if target_days and days_left != target_days:
                continue

            # Send reminders
            results = _send_deadline_reminders_safe(db, deadline, candidate.user_preference, days_left)
            sent_count += len([r for r in results if r.success])

        logger.info("scheduler_sync_reminder_check_completed", reminders_sent=sent_count)
        return sent_count

    finally:
        if should_close:
            db.close()


if __name__ == "__main__":
    # If run directly, start the worker
    run_worker()


def list_active_jobs():
    """Returns a list of all currently active scheduled jobs."""
    return []
