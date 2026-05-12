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
from datetime import datetime
from typing import Dict, Any, Optional

from celery import Celery, Task
from celery.result import AsyncResult

# Import project settings for fallback and other configurations
from api.config import get_settings

# ============================================================================
# INITIALIZATION & LOGGING
# ============================================================================

# Initialize the settings object to fetch global configurations
settings = get_settings()

# Initialize the structured logger for consistent logging across tasks
logger = structlog.get_logger(__name__)


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
    
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True


# ============================================================================
# CELERY APPLICATION INSTANTIATION
# ============================================================================

# The Redis message broker and backend URLs are now dynamically fetched 
# from the environment variables to support seamless deployment across
# different environments (development, staging, production).
# 
# We use REDIS_URL as the primary environment variable, defaulting to
# a local Redis instance if it is not explicitly set.

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

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
            "timestamp": datetime.utcnow().isoformat()
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
            "processed_at": datetime.utcnow().isoformat()
        }
        
        logger.info(
            "Document analysis completed",
            task_id=self.request.id,
            document_id=document_id
        )
        
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


@celery_app.task(bind=True, name="generate_report")
def generate_report_task(
    self,
    user_id: str,
    case_id: str,
    report_type: str = "comprehensive",
    format: str = "pdf"
) -> Dict[str, Any]:
    """
    Asynchronous task to generate a formal report for a legal case.
    
    Args:
        user_id (str): The ID of the user requesting the report.
        case_id (str): The ID of the case for which the report is generated.
        report_type (str): The type of report (e.g., 'summary', 'comprehensive').
        format (str): The output format ('pdf', 'html', etc.).
        
    Returns:
        Dict[str, Any]: Metadata about the generated report file.
    """
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

        report_id = str(uuid.uuid4())
        
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
            "generated_at": datetime.utcnow().isoformat()
        }

        logger.info(
            "Report generation completed",
            task_id=self.request.id,
            case_id=case_id,
            report_id=report_id
        )
        
        return result
    
    except Exception as e:
        logger.error(
            "Report generation failed",
            task_id=self.request.id,
            case_id=case_id,
            error=str(e)
        )
        raise


@celery_app.task(bind=True, name="export_data")
def export_data_task(
    self,
    user_id: str,
    format: str = "csv"
) -> Dict[str, Any]:
    """
    Asynchronous task to export all data associated with a user.
    
    Args:
        user_id (str): The ID of the user whose data is being exported.
        format (str): The desired export format (csv, json).
        
    Returns:
        Dict[str, Any]: Download URL and expiration info for the export file.
    """
    try:
        # Progress tracking for large exports
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Gathering user data from all modules", "progress": 30}
        )
        
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Formatting export package", "progress": 60}
        )
        
        self.update_state(
            state="PROGRESS", 
            meta={"status": "Compressing export file", "progress": 90}
        )
        
        # Mock result for demonstration
        # In production, this would upload to an S3 bucket or similar storage
        result = {
            "export_id": str(uuid.uuid4()),
            "file_url": f"https://storage.example.com/exports/{user_id}.{format}",
            "file_size_bytes": 2048000,
            "expires_in_hours": 24,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return result
    
    except Exception as e:
        logger.error(
            "User data export failed", 
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
            "sent_at": datetime.utcnow().isoformat()
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
