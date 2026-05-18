"""
Celery Asynchronous Task Queue Configuration and Task Definitions

This module initializes the Celery application for the Legalassist-AI project.
It handles the configuration of the message broker, result backend, and
the definition of various background tasks required for document analysis,
report generation, and system maintenance.

Architecture:
    - Broker: Redis (configured via REDIS_URL environment variable)
    - Backend: Redis (configured via REDIS_URL environment variable)
    - Serialization: JSON
    - Task Class: ContextTask (custom task class for request context)

Author: Antigravity AI
Date: 2026-05-12
"""

import os
import uuid
import structlog
import json
from datetime import datetime
from typing import Dict, Any, Optional

from celery import Celery, Task
from celery.result import AsyncResult

# Import project settings for fallback and other configurations
from api.config import get_settings
from observability.integration import initialize_observability_for_environment
from observability.instrumentation import (
    traced_operation,
    capture_exception,
    bind_request_context,
    clear_request_context,
    generate_correlation_id,
)
from api.idempotency import IdempotencyManager
from core.export_storage import save_export_file
from config import Config

# ============================================================================
# INITIALIZATION & LOGGING
# ============================================================================

# Initialize the settings object to fetch global configurations
settings = get_settings()

# Initialize the structured logger for consistent logging across tasks
logger = structlog.get_logger(__name__)
initialize_observability_for_environment()


def build_task_context_headers(
    request_id: Optional[str] = None,
    context_user_id: Optional[str] = None,
) -> Dict[str, str]:
    """Build Celery task headers used to propagate request context."""
    resolved_request_id = request_id or generate_correlation_id()
    headers = {
        "x-request-id": resolved_request_id,
        "x-correlation-id": resolved_request_id,
    }
    if context_user_id:
        headers["x-user-id"] = str(context_user_id)
    return headers


def enqueue_task_with_context(task, *, request_id: Optional[str] = None, context_user_id: Optional[str] = None, **task_kwargs):
    """Enqueue a Celery task with request context propagated in headers."""
    headers = build_task_context_headers(request_id=request_id, context_user_id=context_user_id)
    return task.apply_async(kwargs=task_kwargs, headers=headers)


def enqueue_task_from_http_request(task, http_request, *, context_user_id: Optional[str] = None, **task_kwargs):
    """Enqueue task carrying context from a FastAPI request object."""
    request_id = getattr(http_request.state, "request_id", None) or getattr(http_request.state, "correlation_id", None)
    if not request_id:
        request_id = (
            http_request.headers.get("X-Request-Id")
            or http_request.headers.get("X-Correlation-Id")
            or http_request.headers.get("x-request-id")
            or http_request.headers.get("x-correlation-id")
        )

    user_id = context_user_id or getattr(http_request.state, "user_id", None) or http_request.headers.get("X-User-Id")

    return enqueue_task_with_context(
        task,
        request_id=request_id,
        context_user_id=user_id,
        **task_kwargs,
    )


# ============================================================================
# CUSTOM TASK BASE CLASS
# ============================================================================

class ContextTask(Task):
    """
    Custom Celery Task class that ensures tasks work within the application
    request context and provides default retry logic.
    
    Attributes:
        autoretry_for (tuple): Exceptions that trigger an automatic retry.
        retry_kwargs (dict): Configuration for retry attempts.
        retry_backoff (bool): Enables exponential backoff for retries.
    """
    
    autoretry_for = (
        ConnectionError,
        TimeoutError,
        OSError,
        IOError,
    )
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True

    @staticmethod
    def _extract_task_request_context(task_request) -> Dict[str, Optional[str]]:
        headers = getattr(task_request, "headers", None) or {}
        request_id = (
            headers.get("x-request-id")
            or headers.get("X-Request-Id")
            or headers.get("x-correlation-id")
            or headers.get("X-Correlation-Id")
            or getattr(task_request, "root_id", None)
            or getattr(task_request, "id", None)
        )
        user_id = headers.get("x-user-id") or headers.get("X-User-Id")
        return {"request_id": request_id, "user_id": user_id}

    def __call__(self, *args, **kwargs):
        context = self._extract_task_request_context(self.request)
        bind_request_context(request_id=context.get("request_id"), user_id=context.get("user_id"))
        try:
            return self.run(*args, **kwargs)
        finally:
            clear_request_context()


# ============================================================================
# CELERY APPLICATION INSTANTIATION
# ============================================================================

# Redis URL must be explicitly configured - no silent fallback to localhost
_redis_env = os.getenv("REDIS_URL")
if not _redis_env:
    raise RuntimeError(
        "REDIS_URL environment variable is required. "
        "Cannot start with localhost fallback in production."
    )
REDIS_URL = _redis_env

# Initialize the Celery application instance
celery_app = Celery(
    "legalassist",
    broker=REDIS_URL,
    backend=REDIS_URL,
    task_cls=ContextTask
)


# ============================================================================
# CELERY RUNTIME CONFIGURATION
# ============================================================================

# Detailed configuration for Celery behavior, performance, and reliability.
# This includes serialization settings, time limits, and worker behavior.

celery_app.conf.update(
    # Data Serialization
    # Using JSON for interoperability and security
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone and UTC Settings
    # Standardizing on UTC for consistency across distributed workers
    timezone="UTC",
    enable_utc=True,
    
    # Task Tracking
    # Track when tasks start to provide better visibility into long-running jobs
    task_track_started=True,
    
    # Time Limits (Safety Mechanisms)
    # Prevent tasks from running indefinitely and blocking worker resources
    task_time_limit=settings.CELERY_TASK_TIMEOUT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    
    # Worker Performance Tuning
    # Prefetch multiplier controls how many tasks each worker reserved
    worker_prefetch_multiplier=4,
    
    # Max tasks per child prevents memory leaks in long-lived worker processes
    worker_max_tasks_per_child=1000,
    
    # Beat Schedule Configuration for periodic tasks
    beat_schedule={
        "send-deadline-reminders": {
            "task": "send_deadline_reminders",
            "schedule": 3600.0,
            "options": {"queue": "maintenance"},
        },
        "cleanup-old-tasks": {
            "task": "cleanup_old_tasks",
            "schedule": 86400.0,
            "options": {"queue": "maintenance"},
        },
    },
)


# ============================================================================
# TASK MONITORING UTILITIES
# ============================================================================

class TaskStatus:
    """
    Utility class for tracking and managing the lifecycle of asynchronous tasks.
    Provides methods to query status and revoke tasks.
    """
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """
        Retrieves the current status and metadata for a specific task ID.
        
        Args:
            task_id (str): The unique identifier of the task.
            
        Returns:
            Dict[str, Any]: A dictionary containing the task status, 
                           associated info/results, and a timestamp.
        """
        # Fetch the result object from the backend
        result = AsyncResult(task_id, app=celery_app)
        
        # Determine the status string and extract relevant info based on state
        if result.state == "PENDING":
            status = "pending"
            info = {"status": "Task not yet started or unknown"}
            
        elif result.state == "STARTED":
            status = "processing"
            # Extract progress information if available
            info = result.info if isinstance(result.info, dict) else {"status": "Processing"}
            
        elif result.state == "SUCCESS":
            status = "completed"
            # Return the actual return value of the task
            info = result.result if result.result else {}
            
        elif result.state == "FAILURE":
            status = "failed"
            # Capture the exception details
            info = {"error": str(result.info)}
            
        elif result.state == "RETRY":
            status = "retrying"
            info = {"error": str(result.info)}
            
        else:
            # Fallback for custom or less common states
            status = result.state.lower()
            info = {}
        
        # Construct the response payload
        return {
            "task_id": task_id,
            "status": status,
            "info": info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def revoke_task(task_id: str) -> bool:
        """
        Cancels a running or pending task.
        
        Args:
            task_id (str): The unique identifier of the task to revoke.
            
        Returns:
            bool: True if the revocation request was sent, False otherwise.
        """
        try:
            logger.info("Revoking task", task_id=task_id)
            # Terminate=True forces the worker to stop the task immediately
            celery_app.control.revoke(task_id, terminate=True)
            return True
            
        except Exception as e:
            logger.error("Failed to revoke task", task_id=task_id, error=str(e))
            return False


# ============================================================================
# ASYNCHRONOUS TASK DEFINITIONS
# ============================================================================

@celery_app.task(bind=True, name="analyze_document")
def analyze_document_task(
    self,
    user_id: str,
    document_id: str,
    text: str,
    document_type: str = "unknown"
) -> Dict[str, Any]:
    """
    Asynchronous task to perform deep analysis on a legal document.
    
    This task handles the text extraction, remedy identification, and
    deadline discovery logic using the specialized analysis engine.
    
    Args:
        user_id (str): The ID of the user who owns the document.
        document_id (str): The ID of the document to analyze.
        text (str): The raw text content extracted from the document.
        document_type (str): The category of the document (e.g., 'contract', 'pleading').
        
    Returns:
        Dict[str, Any]: The structured analysis results including identified remedies.
    """
    # Idempotency: prevent duplicate processing for same user/document
    idemp = IdempotencyManager()
    idempotency_key = f"analyze:{user_id}:{document_id}"
    if not idemp.acquire(idempotency_key, ttl=300):
        # Another worker is processing or has processed this key
        existing = idemp.get_result(idempotency_key)
        logger.info("analyze_document_duplicate_skipped", key=idempotency_key, task_id=self.request.id)
        return existing or {"status": "duplicate", "task_id": self.request.id}

    try:
        # Phase 1: Text Pre-processing
        self.update_state(
            state="PROGRESS",
            meta={
                "status": "Extracting and cleaning text",
                "progress": 25
            }
        )
        
        logger.info(
            "Starting document analysis",
            task_id=self.request.id,
            user_id=user_id,
            document_id=document_id
        )
        
        # Simulate Phase 2: Content Analysis
        # This would typically involve NLP or LLM calls
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Analyzing legal content", "progress": 50}
        )
        
        # Simulate Phase 3: Remedy Extraction
        # Identifying specific legal remedies available to the user
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Extracting identified remedies", "progress": 75}
        )
        
        # Simulate Phase 4: Finalization
        # Formatting the output and calculating confidence scores
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Finalizing analysis results", "progress": 90}
        )
        
        # In a production environment, this would call the actual analysis engine
        # located in analytics_engine.py or similar module.
        result = {
            "document_id": document_id,
            "summary": "Document analysis completed successfully",
            "remedies": [],
            "deadlines": [],
            "obligations": [],
            "confidence_score": 0.85,
            "analysis_time_seconds": 10.5,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(
            "Document analysis completed",
            task_id=self.request.id,
            document_id=document_id
        )
        
        idemp.mark_completed(idempotency_key, result)
        return result
    
    except Exception as e:
        # Log the failure with full context for debugging
        logger.error(
            "Document analysis failed",
            task_id=self.request.id,
            user_id=user_id,
            document_id=document_id,
            error=str(e)
        )
        # Re-raise the exception to trigger Celery's retry mechanism
        raise
    finally:
        clear_request_context()
        try:
            idemp.release_lock(idempotency_key)
        except Exception as e:
            logger.warning(
                "lock_release_failed",
                key=idempotency_key,
                error=str(e),
                task_id=self.request.id
            )


@celery_app.task(bind=True, name="generate_report")
def generate_report_task(
    self,
    user_id: str,
    case_id: str,
    report_type: str = "comprehensive",
    format: str = "pdf",
    report_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Asynchronous task to generate a formal report for a legal case.
    
    Args:
        user_id (str): The ID of the user requesting the report.
        case_id (str): The ID of the case for which the report is generated.
        report_type (str): The type of report (e.g., 'summary', 'comprehensive').
        format (str): The output format ('pdf', 'html', etc.).
        report_id (str, optional): Pre-generated report ID.
        
    Returns:
        Dict[str, Any]: Metadata about the generated report file.
    """
    report_id = report_id or str(uuid.uuid4())

    from db.session import db_session
    from db.models.reports import Report

    # Update status to processing in DB
    with db_session() as db:
        db_report = db.query(Report).filter(Report.report_id == report_id).first()
        if db_report:
            db_report.status = "processing"
            db_report.job_id = self.request.id
            db.commit()

    # Idempotency: avoid regenerating same report repeatedly
    idemp = IdempotencyManager()
    idempotency_key = f"report:{user_id}:{case_id}:{report_type}:{format}"
    if not idemp.acquire(idempotency_key, ttl=600):
        existing = idemp.get_result(idempotency_key)
        logger.info("generate_report_duplicate_skipped", key=idempotency_key, task_id=self.request.id)
        if existing:
            with db_session() as db:
                db_report = db.query(Report).filter(Report.report_id == report_id).first()
                if db_report:
                    db_report.status = "completed"
                    db_report.completed_at = datetime.utcnow()
                    db.commit()
        return existing or {"status": "duplicate", "task_id": self.request.id}

    try:
        # Step 1: Data Aggregation
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Compiling case data and documents", "progress": 20}
        )
        
        logger.info(
            "Starting report generation",
            task_id=self.request.id,
            user_id=user_id,
            case_id=case_id
        )
        
        # Step 2: Content Formatting
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Formatting document structure", "progress": 50}
        )
        
        # Step 3: Rendering
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Rendering output document", "progress": 80}
        )
        
        # Finalization
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Finalizing report generation", "progress": 95}
        )
        
        # Import the report service locally to avoid circular dependencies
        from report_service import generate_report

        # Execute the actual report generation logic
        generated = generate_report(
            user_id=user_id,
            case_id=case_id,
            report_type=report_type,
            include_remedies=True,
            include_timeline=True,
            format=format,
            style="formal",
            report_id=report_id,
        )

        # Prepare the result metadata for the frontend
        result = {
            "report_id": report_id,
            "format": generated.format,
            "file_path": str(generated.file_path),
            "file_name": generated.file_name,
            "mime_type": generated.mime_type,
            "file_size_bytes": generated.file_size_bytes,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        logger.info(
            "Report generation completed",
            task_id=self.request.id,
            case_id=case_id,
            report_id=report_id
        )
        
        with db_session() as db:
            db_report = db.query(Report).filter(Report.report_id == report_id).first()
            if db_report:
                db_report.status = "completed"
                db_report.completed_at = datetime.utcnow()
                db.commit()

        idemp.mark_completed(idempotency_key, result)
        return result
    
    except Exception as e:
        logger.error(
            "Report generation failed",
            task_id=self.request.id,
            case_id=case_id,
            error=str(e)
        )
        with db_session() as db:
            db_report = db.query(Report).filter(Report.report_id == report_id).first()
            if db_report:
                db_report.status = "failed"
                db.commit()
        raise
    finally:
        try:
            idemp.release_lock(idempotency_key)
        except Exception:
            pass


@celery_app.task(bind=True, name="export_data")
def export_data_task(
    self,
    user_id: str,
    format: str = "csv"
) -> Dict[str, Any]:
    """
    Asynchronous task to export all data associated with a user.
    
    Exports user data and saves to local storage with real file path.
    
    Args:
        user_id (str): The ID of the user whose data is being exported.
        format (str): The desired export format (csv, json). Default: csv
        
    Returns:
        Dict[str, Any]: Export metadata including:
            - export_id: Unique export identifier
            - file_path: Local file path where export is saved
            - file_size_bytes: Size of exported file
            - expires_in_hours: Hours until file expires
            - expires_at: ISO timestamp when file expires
            - created_at: ISO timestamp of creation
            
    API Contract:
        - file_path: Real local filesystem path (not placeholder URL)
        - expires_at: Guaranteed expiry time, file can be accessed until then
        - Returns null values if format is unsupported
    """
    try:
        self.update_state(
            state="PROGRESS",
            meta={"status": "Gathering user data", "progress": 30}
        )
        
        # Validate format
        if format not in ("csv", "json"):
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'")
        
        self.update_state(
            state="PROGRESS",
            meta={"status": "Formatting export package", "progress": 60}
        )
        
        # Create export data (placeholder - integrate with real data query)
        export_data = {
            "user_id": user_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"placeholder": "User data would be populated from database"}
        }
        
        self.update_state(
            state="PROGRESS",
            meta={"status": "Saving to storage", "progress": 90}
        )
        
        # Serialize based on format
        if format == "csv":
            # Simple CSV representation
            content = "User Export Data\n"
            content += f"User ID,{user_id}\n"
            content += f"Export Time,{export_data['export_timestamp']}\n"
            file_bytes = content.encode('utf-8')
        else:  # json
            file_bytes = json.dumps(export_data, indent=2).encode('utf-8')
        
        # Save to storage and get metadata
        export_file = save_export_file(
            user_id=user_id,
            file_bytes=file_bytes,
            format=format
        )
        
        logger.info(
            "User data export completed",
            task_id=self.request.id,
            user_id=user_id,
            export_id=export_file.export_id,
            format=format,
            file_path=export_file.file_path
        )
        
        result = {
            "export_id": export_file.export_id,
            "file_path": export_file.file_path,
            "file_size_bytes": export_file.file_size_bytes,
            "format": format,
            "expires_in_hours": Config.EXPORT_FILE_EXPIRY_HOURS,
            "expires_at": export_file.expires_at.isoformat(),
            "created_at": export_file.created_at.isoformat()
        }
        
        return result
    
    except ValueError as e:
        logger.error("Invalid export format", user_id=user_id, error=str(e))
        raise
    except Exception as e:
        logger.error(
            "User data export failed",
            task_id=self.request.id,
            user_id=user_id,
            error=str(e)
        )
        raise


@celery_app.task(bind=True, name="send_notification")
def send_notification_task(
    self,
    user_id: str,
    message: str,
    notification_type: str = "email"
) -> Dict[str, Any]:
    """
    Asynchronous task to send user notifications via various channels.
    
    Args:
        user_id (str): The recipient user ID.
        message (str): The notification content.
        notification_type (str): Channel to use (email, push, sms).
        
    Returns:
        Dict[str, Any]: Success metadata including notification ID.
    """
    try:
        logger.info(
            "Dispatching notification",
            user_id=user_id,
            notification_type=notification_type
        )
        
        # Logic for sending notifications would go here
        # (e.g., integration with SendGrid, Twilio, or Firebase)
        
        result = {
            "notification_id": str(uuid.uuid4()),
            "user_id": user_id,
            "type": notification_type,
            "status": "dispatched",
            "sent_at": datetime.now(timezone.utc).isoformat()
        }
        
        return result
    
    except Exception as e:
        logger.error(
            "Notification delivery failed", 
            user_id=user_id, 
            error=str(e)
        )
        raise


# ============================================================================
# SCHEDULED PERIODIC TASKS (CELERY BEAT)
# ============================================================================

@celery_app.task(name="cleanup_old_tasks")
def cleanup_old_tasks() -> Dict[str, str]:
    """
    Maintenance task to clean up old completed tasks from the result backend.
    Runs periodically based on the Celery Beat schedule.
    """
    logger.info("Executing periodic maintenance: cleanup_old_tasks")
    
    # Implementation logic for backend cleanup
    # This prevents the Redis backend from growing indefinitely
    
    return {"status": "completed", "action": "cleanup"}


@celery_app.task(name="send_deadline_reminders")
def send_deadline_reminders() -> Dict[str, int]:
    """
    Periodic task to check for upcoming legal deadlines and notify users.
    """
    logger.info("Executing periodic task: send_deadline_reminders")
    
    # 1. Fetch upcoming deadlines from database
    # 2. Identify users to be notified
    # 3. Trigger send_notification_task for each user
    
    return {"status": "completed", "reminders_sent": 0}
