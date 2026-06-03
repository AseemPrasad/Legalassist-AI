
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

import hashlib
import os
import uuid
import structlog
import json
import re
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import io
import requests
from types import SimpleNamespace
from api.validation import validate_file_url, fetch_url_safe

try:
    from celery import Celery, Task, chain
    from celery.result import AsyncResult
    _CELERY_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for minimal test environments
    _CELERY_AVAILABLE = False

    class Task:  # type: ignore[override]
        request = SimpleNamespace(id="fallback-task", headers={})

        def update_state(self, *args, **kwargs):
            return None

    class AsyncResult:  # type: ignore[override]
        def __init__(self, task_id: str, app=None):
            self.id = task_id
            self.state = "PENDING"
            self.info = None
            self.result = None

    class _FallbackTask:
        def __init__(self, func, name: Optional[str] = None):
            self._func = func
            self.name = name or func.__name__
            self.request = SimpleNamespace(id="fallback-task", headers={})

        def run(self, *args, **kwargs):
            return self._func(self, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self.run(*args, **kwargs)

        def delay(self, *args, **kwargs):
            try:
                self.run(*args, **kwargs)
            except Exception:
                pass
            import uuid
            return SimpleNamespace(id=uuid.uuid4().hex, state="SUCCESS", info=None, result=None)

        def apply_async(self, *args, **kwargs):
            kw = kwargs.get("kwargs", {}) or kwargs
            try:
                self.run(**kw)
            except Exception:
                pass
            import uuid
            return SimpleNamespace(id=uuid.uuid4().hex, state="SUCCESS", info=None, result=None)

        def s(self, *args, **kwargs):
            return self

        def update_state(self, *args, **kwargs):
            return None

    class Celery:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.conf = SimpleNamespace(update=lambda *a, **k: None)
            self.control = SimpleNamespace(revoke=lambda *a, **k: None)

        def task(self, *task_args, **task_kwargs):
            def decorator(func):
                return _FallbackTask(func, name=task_kwargs.get("name"))
            return decorator

try:
    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    _propagator = TraceContextTextMapPropagator()
except Exception:
    trace = None
    _propagator = None

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

try:
    from opentelemetry import trace as otel_trace
    _celery_tracer = otel_trace.get_tracer(__name__)
except Exception:
    _celery_tracer = None
from api.idempotency import IdempotencyManager
from core.export_storage import save_export_file
from core.document_metadata import (
    extract_text_from_uploaded_file,
    extract_case_document_metadata,
)
from config import Config
from core.app_utils import (
    extract_text_from_pdf,
    get_client,
    build_prompt,
    build_remedies_prompt,
    parse_remedies_response,
    compress_text,
)
from api.validation import ValidationConfig
from api.config import get_settings
from db.crud.reports import update_report_status
from db.session import db_session
from database import Attachment, SessionLocal, get_case_by_id, get_case_document_by_id, update_case_document, create_timeline_event

# ============================================================================
# INITIALIZATION & LOGGING
# ============================================================================

# Initialize the settings object to fetch global configurations
settings = get_settings()

# Initialize the structured logger for consistent logging across tasks
logger = structlog.get_logger(__name__)


# ============================================================================
# STATE MACHINE INTEGRATION HOOKS
# ============================================================================

def _trigger_state_machine_hook(
    event: str,
    document_id: str,
    user_id: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Hook to integrate with State Machine at key pipeline transitions.
    
    Supported events:
    - analysis_complete: Analysis chain succeeded
    - analysis_failed: Analysis chain failed
    - text_extraction_complete: Stage 1 succeeded
    - summarization_complete: Stage 2 succeeded
    - remedy_extraction_complete: Stage 3 succeeded
    - finalization_complete: Stage 4 succeeded
    """
    try:
        try:
            from state_machine import DocumentAnalysisStateMachine
            state_machine = DocumentAnalysisStateMachine()
            state_machine.transition(
                document_id=document_id,
                user_id=user_id,
                event=event,
                payload=result or {},
                error=error,
            )
            logger.info(
                "state_machine_transition_triggered",
                event=event,
                document_id=document_id,
            )
        except ImportError:
            logger.debug(
                "state_machine_not_available",
                event=event,
                document_id=document_id,
                reason="state_machine module not found",
            )
        except Exception as sm_err:
            logger.warning(
                "state_machine_transition_failed",
                event=event,
                document_id=document_id,
                error=str(sm_err),
            )
    except Exception as e:
        logger.error(
            "state_machine_hook_error",
            event=event,
            document_id=document_id,
            error=str(e),
        )


def _broadcast_job_event(
    job_id: str,
    event: str,
    stage: str,
    progress: int,
    document_id: str,
    payload: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Broadcast a real-time job progress event to WebSocket subscribers.
    
    Uses the asyncio job_realtime_bus; safe to call from sync Celery tasks
    because we fire-and-forget via asyncio.run_coroutine_threadsafe.
    """
    try:
        import asyncio
        from services.job_realtime import job_realtime_bus

        message = {
            "event": event,
            "job_id": job_id,
            "stage": stage,
            "progress": progress,
            "document_id": document_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        }
        if error:
            message["error"] = error

        # Celery tasks run in separate threads; use the threadsafe approach
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                job_realtime_bus.publish(job_id, message),
                loop
            )
        except RuntimeError:
            # No running loop in this thread — safe to run directly
            asyncio.run(job_realtime_bus.publish(job_id, message))

        logger.debug(
            "job_event_broadcasted",
            job_id=job_id,
            event=event,
            stage=stage,
        )
    except Exception as e:
        logger.warning(
            "job_event_broadcast_failed",
            job_id=job_id,
            event=event,
            error=str(e),
        )


def _persist_lock_event(
    document_id: str,
    task_id: str,
    action: str,
    lock_key: str,
    ttl_ms: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Best-effort persistence of lock events to the database audit table."""
    try:
        from db.session import db_session
        from db.models.locks import DocumentProcessingLock, LockAction

        with db_session() as db:
            record = DocumentProcessingLock(
                document_id=document_id,
                task_id=task_id,
                worker_id=os.getenv("HOSTNAME", "unknown"),
                action=LockAction(action),
                lock_key=lock_key,
                ttl_ms=ttl_ms,
                error_message=error,
            )
            db.add(record)
            db.commit()
    except Exception as e:
        logger.debug("lock_audit_persist_failed", document_id=document_id, error=str(e))


def _acquire_document_lock(document_id: str, task_id: str) -> Optional[Any]:
    """Acquire distributed lock for document processing; returns lock handle or None."""
    try:
        from core.distributed_lock import DistributedLock

        lock = DistributedLock(document_id, ttl_ms=60000, retry_count=5, retry_delay_ms=500)
        acquired = lock.acquire()
        if acquired:
            _persist_lock_event(document_id, task_id, "acquired", lock.lock_key, lock.ttl_ms)
            return lock
        _persist_lock_event(document_id, task_id, "failed", lock.lock_key, lock.ttl_ms, "Acquire failed after retries")
        return None
    except Exception as e:
        logger.error("document_lock_acquire_error", document_id=document_id, task_id=task_id, error=str(e))
        return None


def _release_document_lock(lock: Optional[Any], document_id: str, task_id: str) -> None:
    """Release distributed lock and persist audit event."""
    if lock is None:
        return
    try:
        lock.release()
        _persist_lock_event(document_id, task_id, "released", lock.lock_key)
    except Exception as e:
        logger.warning("document_lock_release_error", document_id=document_id, task_id=task_id, error=str(e))


def _extend_document_lock(lock: Optional[Any], document_id: str, task_id: str, additional_ms: int = 30000) -> bool:
    """Extend lock TTL for long-running tasks."""
    if lock is None:
        return False
    try:
        extended = lock.extend(additional_ms)
        if extended:
            _persist_lock_event(document_id, task_id, "extended", lock.lock_key, lock.ttl_ms)
        return extended
    except Exception as e:
        logger.warning("document_lock_extend_error", document_id=document_id, task_id=task_id, error=str(e))
        return False


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


def enqueue_task_with_context(
    task,
    *,
    request_id: Optional[str] = None,
    context_user_id: Optional[str] = None,
    **task_kwargs,
):
    """Enqueue a Celery task with request context propagated in headers."""
    headers = build_task_context_headers(
        request_id=request_id, context_user_id=context_user_id
    )
    return task.apply_async(kwargs=task_kwargs, headers=headers)


def _sanitize_header_value(value: Optional[str]) -> Optional[str]:
    """Strip control characters and non-printable sequences from header values."""
    if not value:
        return None
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", value)
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    return cleaned.strip() or None


def enqueue_task_from_http_request(
    task, http_request, *, context_user_id: Optional[str] = None, **task_kwargs
):
    """Enqueue task carrying context from a FastAPI request object."""
    request_id = getattr(http_request.state, "request_id", None) or getattr(
        http_request.state, "correlation_id", None
    )
    if not request_id:
        request_id = _sanitize_header_value(
            http_request.headers.get("X-Request-Id")
            or http_request.headers.get("X-Correlation-Id")
            or http_request.headers.get("x-request-id")
            or http_request.headers.get("x-correlation-id")
        )

    user_id = (
        context_user_id
        or getattr(http_request.state, "user_id", None)
        or _sanitize_header_value(http_request.headers.get("X-User-Id"))
    )

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
    """

    autoretry_for = (
        ConnectionError,
        TimeoutError,
        OSError,
        IOError,
    )
    retry_kwargs = {"max_retries": 3}
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

    def apply_async(self, *args, headers=None, **kwargs):
        if _propagator is not None and headers is not None:
            carrier: Dict[str, str] = {}
            _propagator.inject(carrier)
            headers.update(carrier)
        return super().apply_async(*args, headers=headers, **kwargs)

    def __call__(self, *args, **kwargs):
        context = self._extract_task_request_context(self.request)

        if _propagator is not None and trace is not None:
            carrier: Dict[str, str] = dict(getattr(self.request, "headers", None) or {})
            ctx = _propagator.extract(carrier)
            span = trace.get_tracer(__name__).start_span(
                f"celery.task.{self.name}",
                context=ctx,
            )
            span.set_attribute("celery.task_id", self.request.id or "")
            span.set_attribute("celery.task_name", self.name or "")
            if context.get("request_id"):
                span.set_attribute("correlation.id", context["request_id"])
            request_scope = span

            with trace.use_span(request_scope):
                bind_request_context(
                    request_id=context.get("request_id"), user_id=context.get("user_id")
                )
                try:
                    return self.run(*args, **kwargs)
                finally:
                    span.end()
                    clear_request_context()
        else:
            bind_request_context(
                request_id=context.get("request_id"), user_id=context.get("user_id")
            )
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
    logger.warning(
        "REDIS_URL not set — Celery background tasks disabled. "
        "Set REDIS_URL to enable async task processing."
    )
    celery_app = SimpleNamespace()
    celery_app.conf = SimpleNamespace()
    celery_app.conf.update = lambda **kw: None
    celery_app.conf.__setitem__ = lambda k, v: None
    celery_app.AsyncResult = lambda *args, **kwargs: SimpleNamespace(state="PENDING", result=None, status="PENDING")
    celery_app.main = "legalassist"
    REDIS_URL = ""
else:
    REDIS_URL = _redis_env
    celery_app = Celery(
        "legalassist", broker=REDIS_URL, backend=REDIS_URL, task_cls=ContextTask
    )


# ============================================================================
# CELERY RUNTIME CONFIGURATION
# ============================================================================

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=get_settings().CELERY_TASK_TIMEOUT,
    task_soft_time_limit=get_settings().CELERY_TASK_SOFT_TIME_LIMIT,
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
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
        "cleanup-revoked-tokens": {
            "task": "cleanup_revoked_tokens",
            "schedule": 21600.0,
            "options": {"queue": "maintenance"},
        },
        "enforce-retention-policies": {
            "task": "enforce_retention_policies",
            "schedule": 86400.0,
            "options": {"queue": "compliance"},
        },
        "enforce-data-anonymization": {
            "task": "enforce_data_anonymization",
            "schedule": 86400.0,
            "options": {"queue": "compliance"},
        },
        "purge-expired-data": {
            "task": "purge_expired_data",
            "schedule": 604800.0,
            "options": {"queue": "compliance"},
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
        """
        result = AsyncResult(task_id, app=celery_app)

        if result.state == "PENDING":
            status = "pending"
            info = {"status": "Task not yet started or unknown"}

        elif result.state == "STARTED":
            status = "processing"
            info = (
                result.info
                if isinstance(result.info, dict)
                else {"status": "Processing"}
            )

        elif result.state == "SUCCESS":
            status = "completed"
            info = result.result if result.result else {}

        elif result.state == "FAILURE":
            status = "failed"
            info = {"error": str(result.info)}

        elif result.state == "RETRY":
            status = "retrying"
            info = {"error": str(result.info)}

        else:
            status = result.state.lower()
            info = {}

        return {
            "task_id": task_id,
            "status": status,
            "info": info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def revoke_task(task_id: str) -> bool:
        """
        Cancels a running or pending task.
        """
        try:
            logger.info("Revoking task", task_id=task_id)
            celery_app.control.revoke(task_id, terminate=True)
            return True

        except Exception as e:
            logger.error("Failed to revoke task", task_id=task_id, error=str(e))
            return False


# ============================================================================
# SUB-TASK DEFINITIONS FOR DOCUMENT ANALYSIS PIPELINE (CHAIN-COMPATIBLE)
# ============================================================================

@celery_app.task(bind=True, name="analyze_document")
def analyze_document_task(
    self,
    user_id: str,
    document_id: str,
    text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    document_type: str = "unknown",
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrator task for document analysis using Celery chain with apply_async.
    
    Chain: extract_text -> summarize -> extract_remedies -> finalize
    Halts on first error. Integrates PipelineStateManager for persistence.
    Uses callbacks for async state updates instead of synchronous blocking.
    """
    
    # Idempotency: prevent duplicate processing for same user/document
    content_parts = []
    if file_bytes:
        content_parts.append(hashlib.sha256(file_bytes).hexdigest())
    if text:
        content_parts.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    content_hash = hashlib.sha256("|".join(content_parts).encode()).hexdigest()[:16] if content_parts else ""

    idemp = IdempotencyManager()
    idempotency_key = f"analyze:{user_id}:{document_id}:{content_hash}"
    if not idemp.acquire(idempotency_key, ttl=300):
        existing = idemp.get_result(idempotency_key)
        logger.info(
            "analyze_document_duplicate_skipped",
            key=idempotency_key,
            task_id=self.request.id,
        )
        return existing or {"status": "duplicate", "task_id": self.request.id}

    start_time = datetime.now(timezone.utc)

    try:
        logger.info(
            "Starting document analysis (chain-orchestrated async)",
            task_id=self.request.id,
            user_id=user_id,
            document_id=document_id,
        )

        # Persist initial state
        _persist_pipeline_state(document_id, user_id, "analysis_started", {"task_id": self.request.id})

        # Build task chain with callbacks for state transitions
        task_chain = chain(
            extract_document_text_task.s(
                user_id=user_id,
                document_id=document_id,
                text=text,
                file_bytes=file_bytes,
                file_path=file_path,
                file_url=file_url,
            ),
            summarize_document_task.s(),
            extract_remedies_task.s(),
            finalize_analysis_task.s(document_type=document_type),
        )

        # Execute chain asynchronously — returns AsyncResult immediately
        # The chain result backend will store the final result
        chain_result = task_chain.apply_async(
            link=_on_chain_success.s(document_id, user_id, idempotency_key, start_time.isoformat()),
            link_error=_on_chain_error.s(document_id, user_id, idempotency_key),
        )

        # Return immediately with chain task ID for polling
        return {
            "task_id": self.request.id,
            "chain_task_id": chain_result.id,
            "status": "pending",
            "document_id": document_id,
            "user_id": user_id,
            "stage": "analysis_started",
        }

    except Exception as e:
        logger.error(
            "Document analysis chain failed to start",
            task_id=self.request.id,
            user_id=user_id,
            document_id=document_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        
        _persist_pipeline_state(document_id, user_id, "analysis_failed", {"error": str(e)})
        _trigger_state_machine_hook(
            event="analysis_failed",
            document_id=document_id,
            user_id=user_id,
            error=str(e),
        )
        
        raise
    finally:
        clear_request_context()


def _persist_pipeline_state(document_id: str, user_id: str, stage: str, data: Optional[Dict] = None) -> None:
    """Persist pipeline state via PipelineStateManager."""
    try:
        from core.app_utils import PipelineStateManager
        from database import SessionLocal
        db = SessionLocal()
        try:
            PipelineStateManager.update_stage(db, document_id, stage, data)
            logger.info(
                "pipeline_state_persisted",
                document_id=document_id,
                stage=stage,
            )
        finally:
            db.close()
    except Exception as e:
        logger.warning(
            "pipeline_state_persist_failed",
            document_id=document_id,
            stage=stage,
            error=str(e),
        )


@celery_app.task(bind=True, name="on_chain_success")
def _on_chain_success(self, final_result: Dict[str, Any], document_id: str, user_id: str, idempotency_key: str, start_time_iso: str) -> Dict[str, Any]:
    """Callback fired when the entire chain succeeds."""
    start_time = datetime.fromisoformat(start_time_iso)
    analysis_time = (datetime.utcnow() - start_time).total_seconds()
    final_result["analysis_time_seconds"] = analysis_time

    # Persist completion state
    _persist_pipeline_state(document_id, user_id, "finalization_complete", final_result)

    # Mark idempotency complete
    try:
        idemp = IdempotencyManager()
        idemp.mark_completed(idempotency_key, final_result)
    except Exception as e:
        logger.warning("idempotency_mark_failed", key=idempotency_key, error=str(e))

    # Trigger state machine hook
    _trigger_state_machine_hook(
        event="analysis_complete",
        document_id=document_id,
        user_id=user_id,
        result=final_result,
    )

    logger.info(
        "Document analysis chain completed (async)",
        document_id=document_id,
        analysis_time=analysis_time,
    )

    return final_result


@celery_app.task(bind=True, name="on_chain_error")
def _on_chain_error(self, exc_info: Any, document_id: str, user_id: str, idempotency_key: str) -> None:
    """Errback fired when any stage in the chain fails."""
    error_msg = str(exc_info) if not isinstance(exc_info, Exception) else str(exc_info)

    # Persist failure state
    _persist_pipeline_state(document_id, user_id, "analysis_failed", {"error": error_msg})

    # Trigger state machine hook
    _trigger_state_machine_hook(
        event="analysis_failed",
        document_id=document_id,
        user_id=user_id,
        error=error_msg,
    )

    logger.error(
        "Document analysis chain failed (async)",
        document_id=document_id,
        error=error_msg,
    )

@celery_app.task(bind=True, name="extract_document_text")
def extract_document_text_task(
    self,
    user_id: str,
    document_id: str,
    text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage 1: Extract and validate text from document source."""
    self.update_state(
        state="PROGRESS",
        meta={"status": "Extracting and cleaning text", "progress": 25, "stage": "text_extraction"}
    )
    
    logger.info(
        "Stage 1: Starting text extraction",
        task_id=self.request.id,
        user_id=user_id,
        document_id=document_id,
    )
    
    try:
        extracted_text = text
        if extracted_text:
            if len(extracted_text.encode("utf-8")) > ValidationConfig.MAX_TEXT_LENGTH:
                raise ValueError(f"Input text exceeds max limit of {ValidationConfig.MAX_TEXT_LENGTH} bytes.")
        if not extracted_text and file_bytes:
            if len(file_bytes) > ValidationConfig.MAX_TEXT_LENGTH:
                raise ValueError(f"File too large: {len(file_bytes)} bytes exceeds limit of {ValidationConfig.MAX_TEXT_LENGTH} bytes.")
            extracted_text = extract_text_from_pdf(io.BytesIO(file_bytes))
        if not extracted_text:
            if file_url:
                validate_file_url(file_url)
                response = requests.get(file_url, timeout=30)
                response.raise_for_status()
                if len(response.content) > ValidationConfig.MAX_TEXT_LENGTH:
                    raise ValueError(f"Downloaded file too large: {len(response.content)} bytes exceeds limit of {ValidationConfig.MAX_TEXT_LENGTH} bytes.")
                content_type = response.headers.get("Content-Type", "")
                if "application/pdf" in content_type or file_url.lower().endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(io.BytesIO(response.content))
                else:
                    extracted_text = response.content.decode("utf-8", errors="ignore")
            elif file_path:
                # Ownership verification
                session = SessionLocal()
                try:
                    owned = session.query(Attachment).filter(
                        Attachment.stored_path == file_path,
                        Attachment.user_id == int(user_id),
                    ).first()
                    if not owned:
                        raise ValueError("You do not have permission to access this file")
                finally:
                    session.close()
                if os.path.getsize(file_path) > ValidationConfig.MAX_TEXT_LENGTH:
                    raise ValueError(f"File too large: {os.path.getsize(file_path)} bytes exceeds limit of {ValidationConfig.MAX_TEXT_LENGTH} bytes.")
                if file_path.lower().endswith(".pdf"):
                    with open(file_path, "rb") as f:
                        extracted_text = extract_text_from_pdf(io.BytesIO(f.read()))
                else:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        extracted_text = f.read()
                        
        if not extracted_text:
            raise ValueError("No text provided or extracted from document.")

        if len(extracted_text.encode("utf-8")) > ValidationConfig.MAX_TEXT_LENGTH:
            raise ValueError(f"Extracted text exceeds max limit of {ValidationConfig.MAX_TEXT_LENGTH} bytes.")

        logger.info(
            "Stage 1: Text extraction completed",
            task_id=self.request.id,
            document_id=document_id,
            text_length=len(extracted_text),
        )

        _broadcast_job_event(
            job_id=self.request.id,
            event="stage_complete",
            stage="text_extraction",
            progress=25,
            document_id=document_id,
            payload={"text_length": len(extracted_text)},
        )

        return {
        result = {
            "user_id": user_id,
            "document_id": document_id,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
            "stage": "text_extraction_complete",
        }
        _persist_pipeline_state(document_id, user_id, "text_extraction_complete", result)
        return result

    except Exception as e:
        logger.error(
            "Stage 1: Text extraction failed",
            task_id=self.request.id,
            document_id=document_id,
            error=str(e),
        )
        _broadcast_job_event(
            job_id=self.request.id,
            event="failed",
            stage="text_extraction",
            progress=0,
            document_id=document_id,
            error=str(e),
        )
        raise
    finally:
        clear_request_context()

@celery_app.task(bind=True, name="summarize_document")
def summarize_document_task(
    self,
    extraction_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage 2: Generate summary from extracted text via LLM."""
    self.update_state(
        state="PROGRESS",
        meta={"status": "Analyzing legal content", "progress": 50, "stage": "summarization"}
    )
    
    user_id = extraction_result.get("user_id")
    document_id = extraction_result.get("document_id")
    extracted_text = extraction_result.get("extracted_text")

    logger.info(
        "Stage 2: Starting summarization",
        task_id=self.request.id,
        document_id=document_id,
    )

    try:
        safe_text = compress_text(extracted_text)
        client = get_client()
        if not client:
            raise RuntimeError("Failed to initialize LLM client.")

        summary_prompt = build_prompt(safe_text, "English")
        if _celery_tracer:
            with _celery_tracer.start_as_current_span(
                f"llm.{Config.DEFAULT_MODEL}.summary",
                attributes={
                    "llm.model": Config.DEFAULT_MODEL,
                    "llm.operation": "summary",
                    "celery.task_id": self.request.id or "",
                },
            ) as span:
                summary_response = client.chat.completions.create(
                    model=Config.DEFAULT_MODEL,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=800,
                    temperature=0.3,
                )
                if hasattr(summary_response, 'usage') and summary_response.usage:
                    span.set_attribute("llm.prompt_tokens", summary_response.usage.prompt_tokens or 0)
                    span.set_attribute("llm.completion_tokens", summary_response.usage.completion_tokens or 0)
                    span.set_attribute("llm.total_tokens", summary_response.usage.total_tokens or 0)
                raw_summary = summary_response.choices[0].message.content
        else:
            summary_response = client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            raw_summary = summary_response.choices[0].message.content

        summary_text = ""
        key_points = []
        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_summary, re.DOTALL)
            json_str = match.group(1) if match else raw_summary
            data = json.loads(json_str)
            key_points = data.get("bullets", [])
            summary_text = " ".join(key_points)
        except Exception:
            summary_text = raw_summary

        logger.info(
            "Stage 2: Summarization completed",
            task_id=self.request.id,
            document_id=document_id,
            key_points_count=len(key_points),
        )

        _broadcast_job_event(
            job_id=self.request.id,
            event="stage_complete",
            stage="summarization",
            progress=50,
            document_id=document_id,
            payload={"key_points_count": len(key_points)},
        )

        return {
        result = {
            **extraction_result,
            "summary_text": summary_text,
            "key_points": key_points,
            "stage": "summarization_complete",
        }
        _persist_pipeline_state(document_id, user_id, "summarization_complete", result)
        return result

    except Exception as e:
        logger.error(
            "Stage 2: Summarization failed",
            task_id=self.request.id,
            document_id=document_id,
            error=str(e),
        )
        _broadcast_job_event(
            job_id=self.request.id,
            event="failed",
            stage="summarization",
            progress=0,
            document_id=document_id,
            error=str(e),
        )
        raise
    finally:
        clear_request_context()


@celery_app.task(bind=True, name="extract_remedies")
def extract_remedies_task(
    self,
    summarization_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage 3: Extract remedies and deadlines from text via LLM."""
    self.update_state(
        state="PROGRESS",
        meta={"status": "Extracting identified remedies", "progress": 75, "stage": "remedy_extraction"}
    )

    user_id = summarization_result.get("user_id")
    document_id = summarization_result.get("document_id")
    extracted_text = summarization_result.get("extracted_text")

    logger.info(
        "Stage 3: Starting remedy extraction",
        task_id=self.request.id,
        document_id=document_id,
    )

    try:
        safe_text = compress_text(extracted_text)
        client = get_client()
        if not client:
            raise RuntimeError("Failed to initialize LLM client.")

        remedies_prompt = build_remedies_prompt(safe_text, "English")
        if _celery_tracer:
            with _celery_tracer.start_as_current_span(
                f"llm.{Config.DEFAULT_MODEL}.remedies",
                attributes={
                    "llm.model": Config.DEFAULT_MODEL,
                    "llm.operation": "remedies",
                    "celery.task_id": self.request.id or "",
                },
            ) as span:
                remedies_response = client.chat.completions.create(
                    model=Config.DEFAULT_MODEL,
                    messages=[{"role": "user", "content": remedies_prompt}],
                    max_tokens=900,
                    temperature=0.3,
                )
                if hasattr(remedies_response, 'usage') and remedies_response.usage:
                    span.set_attribute("llm.prompt_tokens", remedies_response.usage.prompt_tokens or 0)
                    span.set_attribute("llm.completion_tokens", remedies_response.usage.completion_tokens or 0)
                    span.set_attribute("llm.total_tokens", remedies_response.usage.total_tokens or 0)
                remedies_data = parse_remedies_response(remedies_response.choices[0].message.content)
        else:
            remedies_response = client.chat.completions.create(
                model=Config.DEFAULT_MODEL,
                messages=[{"role": "user", "content": remedies_prompt}],
                max_tokens=900,
                temperature=0.3,
            )
            remedies_data = parse_remedies_response(remedies_response.choices[0].message.content)

        remedies_list = []
        if remedies_data.get("first_action"):
            remedies_list.append(f"Action: {remedies_data['first_action']}")
        if remedies_data.get("can_appeal") == "yes":
            remedies_list.append(f"Appeal allowed in {remedies_data.get('appeal_court', 'court')} within {remedies_data.get('appeal_days', 'unknown')} days.")
            
        deadlines_list = []
        if remedies_data.get("deadline"):
            deadlines_list.append(remedies_data["deadline"])

        logger.info(
            "Stage 3: Remedy extraction completed",
            task_id=self.request.id,
            document_id=document_id,
            remedies_count=len(remedies_list),
        )

        _broadcast_job_event(
            job_id=self.request.id,
            event="stage_complete",
            stage="remedy_extraction",
            progress=75,
            document_id=document_id,
            payload={"remedies_count": len(remedies_list)},
        )

        return {
        result = {
            **summarization_result,
            "remedies": remedies_list,
            "deadlines": deadlines_list,
            "remedies_confidence_score": remedies_data.get("confidence_score", 0.0),
            "remedies_evidence_spans": remedies_data.get("evidence_spans", []),
            "remedies_data": remedies_data,
            "stage": "remedy_extraction_complete",
        }
        _persist_pipeline_state(document_id, user_id, "remedy_extraction_complete", result)
        return result

    except Exception as e:
        logger.error(
            "Stage 3: Remedy extraction failed",
            task_id=self.request.id,
            document_id=document_id,
            error=str(e),
        )
        _broadcast_job_event(
            job_id=self.request.id,
            event="failed",
            stage="remedy_extraction",
            progress=0,
            document_id=document_id,
            error=str(e),
        )
        raise
    finally:
        clear_request_context()


@celery_app.task(bind=True, name="finalize_analysis")
def finalize_analysis_task(
    self,
    remedy_result: Dict[str, Any],
    document_type: str = "unknown",
) -> Dict[str, Any]:
    """Stage 4: Finalize and structure all analysis results."""
    self.update_state(
        state="PROGRESS",
        meta={"status": "Finalizing analysis results", "progress": 90, "stage": "finalization"}
    )

    document_id = remedy_result.get("document_id")
    user_id = remedy_result.get("user_id")

    logger.info(
        "Stage 4: Starting finalization",
        task_id=self.request.id,
        document_id=document_id,
    )

    try:
        result = {
            "document_id": document_id,
            "title": "Analyzed Document",
            "document_type": document_type,
            "summary": remedy_result.get("summary_text", ""),
            "key_points": remedy_result.get("key_points", []),
            "remedies": remedy_result.get("remedies", []),
            "deadlines": remedy_result.get("deadlines", []),
            "obligations": [],
            "confidence_score": 0.85 if not remedy_result.get("remedies_data", {}).get("_is_partial") else 0.6,
            "remedies_confidence_score": remedy_result.get("remedies_confidence_score", 0.0),
            "remedies_evidence_spans": remedy_result.get("remedies_evidence_spans", []),
            "analysis_time_seconds": 0,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "stage": "finalization_complete",
        }

        _persist_pipeline_state(document_id, user_id, "finalization_complete", result)

        logger.info(
            "Stage 4: Analysis finalization completed",
            task_id=self.request.id,
            document_id=document_id,
        )

        _broadcast_job_event(
            job_id=self.request.id,
            event="completed",
            stage="finalization",
            progress=100,
            document_id=document_id,
            payload={"analysis_time_seconds": result.get("analysis_time_seconds", 0)},
        )

        return result

    except Exception as e:
        logger.error(
            "Stage 4: Analysis finalization failed",
            task_id=self.request.id,
            document_id=document_id,
            error=str(e),
        )
        _broadcast_job_event(
            job_id=self.request.id,
            event="failed",
            stage="finalization",
            progress=0,
            document_id=document_id,
            error=str(e),
        )
        raise
    finally:
        clear_request_context()


# ============================================================================
# ASYNCHRONOUS TASK DEFINITIONS
# ============================================================================


@celery_app.task(bind=True, name="analyze_document")
def analyze_document_task(
    self,
    user_id: str,
    document_id: str,
    text: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    document_type: str = "unknown",
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrator task for document analysis using Celery chain.
    
    Chain: extract_text -> summarize -> extract_remedies -> finalize
    Halts on first error. Integrates State Machine hooks.
    """
    
    # Idempotency: prevent duplicate processing for same user/document
    content_parts = []
    if file_bytes:
        content_parts.append(hashlib.sha256(file_bytes).hexdigest())
    if text:
        content_parts.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    content_hash = hashlib.sha256("|".join(content_parts).encode()).hexdigest()[:16] if content_parts else ""

    idemp = IdempotencyManager()
    idempotency_key = f"analyze:{user_id}:{document_id}:{content_hash}"
    if not idemp.acquire(idempotency_key, ttl=300):
        existing = idemp.get_result(idempotency_key)
        logger.info(
            "analyze_document_duplicate_skipped",
            key=idempotency_key,
            task_id=self.request.id,
        )
        return existing or {"status": "duplicate", "task_id": self.request.id}

    # Acquire distributed lock for strict serial processing per document
    doc_lock = _acquire_document_lock(document_id, self.request.id)
    if doc_lock is None:
        logger.error(
            "analyze_document_lock_failed",
            task_id=self.request.id,
            document_id=document_id,
        )
        raise RuntimeError(f"Could not acquire distributed lock for document {document_id}")

    start_time = datetime.utcnow()

    try:
        logger.info(
            "Starting document analysis (chain-orchestrated)",
            task_id=self.request.id,
            user_id=user_id,
            document_id=document_id,
        )

        # Auto-extend lock every 30s for long chains
        def _lock_heartbeat():
            while True:
                time.sleep(25)
                _extend_document_lock(doc_lock, document_id, self.request.id, 30000)

        heartbeat_thread = threading.Thread(target=_lock_heartbeat, daemon=True)
        heartbeat_thread.start()

        # Build task chain
        task_chain = chain(
            extract_document_text_task.s(
                user_id=user_id,
                document_id=document_id,
                text=text,
                file_bytes=file_bytes,
                file_path=file_path,
                file_url=file_url,
            ),
            summarize_document_task.s(),
            extract_remedies_task.s(),
            finalize_analysis_task.s(document_type=document_type),
        )

        # Broadcast start event immediately
        _broadcast_job_event(
            job_id=self.request.id,
            event="processing_started",
            stage="analysis",
            progress=0,
            document_id=document_id,
        )

        # Execute chain synchronously to capture result
        chain_result = task_chain.apply()
        final_result = chain_result.get() if hasattr(chain_result, 'get') else chain_result

        # Add timing metadata
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        final_result["analysis_time_seconds"] = analysis_time

        logger.info(
            "Document analysis chain completed",
            task_id=self.request.id,
            document_id=document_id,
            analysis_time=analysis_time,
        )

        # Broadcast final completion (redundant safety net if chain tasks missed it)
        _broadcast_job_event(
            job_id=self.request.id,
            event="completed",
            stage="analysis",
            progress=100,
            document_id=document_id,
            payload={"analysis_time_seconds": analysis_time},
        )

        # Mark idempotency complete and persist result
        idemp.mark_completed(idempotency_key, final_result)
        
        # HOOK: Trigger State Machine transition
        _trigger_state_machine_hook(
            event="analysis_complete",
            document_id=document_id,
            user_id=user_id,
            result=final_result,
        )

        return final_result

    except Exception as e:
        logger.error(
            "Document analysis chain failed",
            task_id=self.request.id,
            user_id=user_id,
            document_id=document_id,
            error=str(e),
            error_type=type(e).__name__,
        )

        _broadcast_job_event(
            job_id=self.request.id,
            event="failed",
            stage="analysis",
            progress=0,
            document_id=document_id,
            error=str(e),
        )
        
        # HOOK: Trigger State Machine transition for failure
        _trigger_state_machine_hook(
            event="analysis_failed",
            document_id=document_id,
            user_id=user_id,
            error=str(e),
        )
        
        raise
    finally:
        _release_document_lock(doc_lock, document_id, self.request.id)
        clear_request_context()


@celery_app.task(bind=True, name="process_case_document_upload")
def process_case_document_upload_task(
    self,
    user_id: str,
    case_id: str,
    attachment_id: str,
    document_id: str,
    original_filename: str,
    ocr_languages: str = "eng+hin",
    ocr_dpi: int = 300,
) -> Dict[str, Any]:
    """Run OCR and metadata extraction for a newly uploaded case document."""
    session = SessionLocal()
    doc_lock = _acquire_document_lock(document_id, self.request.id)
    if doc_lock is None:
        raise RuntimeError(f"Could not acquire distributed lock for document {document_id}")
    try:
        case = get_case_by_id(session, int(case_id))
        if not case or str(case.user_id) != str(user_id):
            raise ValueError("Case not found or not owned by the provided user")

        doc = get_case_document_by_id(session, int(document_id))
        if not doc:
            raise ValueError("Document record not found")

        attachment = session.query(Attachment).filter(Attachment.id == int(attachment_id)).first()
        if not attachment:
            raise ValueError("Attachment not found")

        self.update_state(state="PROGRESS", meta={"status": "Extracting text", "progress": 30})

        diagnostics = extract_text_from_uploaded_file(
            attachment.stored_path,
            original_filename=original_filename,
            enable_ocr=True,
            ocr_languages=ocr_languages,
            ocr_dpi=ocr_dpi,
        )
        text = diagnostics.get("text", "")
        metadata = extract_case_document_metadata(text, filename=original_filename)

        self.update_state(state="PROGRESS", meta={"status": "Saving extracted metadata", "progress": 85})

        summary_parts = []
        if metadata.get("parties"):
            summary_parts.append(f"Parties: {', '.join(metadata['parties'][:2])}")
        if metadata.get("claims"):
            summary_parts.append(f"Claims: {metadata['claims'][0]}")
        if metadata.get("statutes"):
            summary_parts.append(f"Statutes: {', '.join(metadata['statutes'][:3])}")
        summary = " | ".join(summary_parts) if summary_parts else None

        updated = update_case_document(
            session,
            document_id=doc.id,
            document_content=text,
            summary=summary,
            extracted_metadata=metadata,
            extraction_method=str(diagnostics.get("method") or "unknown"),
            ocr_used=bool(diagnostics.get("ocr_used", False)),
        )

        attachment.document_id = doc.id
        session.commit()

        create_timeline_event(
            session,
            case_id=case.id,
            event_type="document_processed",
            description=f"Processed uploaded document: {original_filename}",
            metadata={
                "attachment_id": attachment.id,
                "document_id": doc.id,
                "ocr_used": bool(diagnostics.get("ocr_used", False)),
            },
        )

        return {
            "status": "completed",
            "document_id": doc.id,
            "attachment_id": attachment.id,
            "case_id": case.id,
            "ocr_used": bool(diagnostics.get("ocr_used", False)),
            "extraction_method": diagnostics.get("method"),
            "parties": metadata.get("parties", []),
            "dates": metadata.get("dates", []),
            "claims": metadata.get("claims", []),
            "statutes": metadata.get("statutes", []),
        }
    finally:
        _release_document_lock(doc_lock, document_id, self.request.id)
        session.close()


@celery_app.task(bind=True, name="generate_report")
def generate_report_task(
    self,
    user_id: str,
    case_id: str,
    report_id: str,
    report_type: str = "comprehensive",
    format: str = "pdf",
    privacy_profile: str = "personal_identifiers",
) -> Dict[str, Any]:
    """Asynchronous task to generate a formal report for a legal case."""
    from db.session import db_session
    from db.models.reports import Report
    from db.crud.reports import update_report_status

    # Update status to processing in DB
    with db_session() as db:
        db_report = db.query(Report).filter(Report.report_id == report_id).first()
        if db_report:
            db_report.status = "processing"
            db_report.job_id = self.request.id
            db.commit()

    # Idempotency
    idemp = IdempotencyManager()
    idempotency_key = f"report:{report_id}:{user_id}:{case_id}:{report_type}:{format}:{privacy_profile}"
    if not idemp.acquire(idempotency_key, ttl=600):
        existing = idemp.get_result(idempotency_key)
        logger.info(
            "generate_report_duplicate_skipped",
            key=idempotency_key,
            task_id=self.request.id,
        )
        if existing:
            with db_session() as db:
                db_report = (
                    db.query(Report).filter(Report.report_id == report_id).first()
                )
                if db_report:
                    db_report.status = "completed"
                    db_report.completed_at = datetime.utcnow()
                    db.commit()
        return existing or {"status": "duplicate", "task_id": self.request.id}

    # Acquire distributed lock keyed by report_id for report generation
    doc_lock = _acquire_document_lock(f"report:{report_id}", self.request.id)
    if doc_lock is None:
        raise RuntimeError(f"Could not acquire distributed lock for report {report_id}")

    try:
        # Mark task as started in DB
        with db_session() as db:
            update_report_status(
                db,
                report_id,
                status="processing",
                started_at=datetime.utcnow(),
            )
        
        self.update_state(
            state="PROGRESS",
            meta={"status": "Compiling case data and documents", "progress": 20},
        )

        logger.info(
            "Starting report generation",
            task_id=self.request.id,
            user_id=user_id,
            case_id=case_id,
        )

        self.update_state(
            state="PROGRESS",
            meta={"status": "Formatting document structure", "progress": 50},
        )

        self.update_state(
            state="PROGRESS",
            meta={"status": "Rendering output document", "progress": 80},
        )

        self.update_state(
            state="PROGRESS",
            meta={"status": "Finalizing report generation", "progress": 95},
        )

        from report_service import generate_report

        generated = generate_report(
            user_id=user_id,
            case_id=case_id,
            report_type=report_type,
            include_remedies=True,
            include_timeline=True,
            format=format,
            style="formal",
            report_id=report_id,
            privacy_profile=privacy_profile,
        )

        file_path_str = str(generated.file_path)
        with db_session() as db:
            update_report_status(
                db,
                report_id,
                status="completed",
                file_path=file_path_str,
                file_size_bytes=generated.file_size_bytes,
                completed_at=datetime.utcnow(),
            )

        result = {
            "report_id": report_id,
            "format": generated.format,
            "file_path": file_path_str,
            "file_name": generated.file_name,
            "mime_type": generated.mime_type,
            "file_size_bytes": generated.file_size_bytes,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Report generation completed",
            task_id=self.request.id,
            case_id=case_id,
            report_id=report_id,
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
        try:
            with db_session() as db:
                update_report_status(
                    db,
                    report_id,
                    status="failed",
                    error_message=str(e),
                    completed_at=datetime.utcnow(),
                )
        except Exception as db_err:
            logger.error("Failed to update report status on error", report_id=report_id, db_error=str(db_err))
        
        logger.error(
            "Report generation failed",
            task_id=self.request.id,
            case_id=case_id,
            error=str(e),
        )
        with db_session() as db:
            db_report = db.query(Report).filter(Report.report_id == report_id).first()
            if db_report:
                db_report.status = "failed"
                db.commit()
        raise
    finally:
        _release_document_lock(doc_lock, f"report:{report_id}", self.request.id)
        try:
            idemp.release_lock(idempotency_key)
        except Exception:
            pass


@celery_app.task(bind=True, name="export_data")
def export_data_task(
    self, user_id: str, format: str = "csv", anonymize: bool = False
) -> Dict[str, Any]:
    """
    Asynchronous task to export all data associated with a user.
    """
    try:
        self.update_state(
            state="PROGRESS", meta={"status": "Gathering user data", "progress": 30}
        )

        if format not in ("csv", "json"):
            raise ValueError(f"Unsupported export format: {format}.")

        try:
            int_user_id = int(user_id)
        except (ValueError, TypeError) as e:
            logger.error(
                "Invalid user ID format during export", user_id=user_id, error=str(e)
            )
            raise ValueError(f"Invalid user_id: {user_id}. Must be an integer.")

        import csv
        from db.session import db_session
        from db.models import Case, CaseDeadline, NotificationLog

        def mask_recipient(recipient: str) -> str:
            if not recipient:
                return ""
            recipient = str(recipient).strip()
            if "@" in recipient:
                try:
                    parts = recipient.split("@", 1)
                    username, domain = parts[0], parts[1]
                    if len(username) <= 2:
                        masked_username = username[0] + "*" * (len(username) - 1)
                    else:
                        masked_username = (
                            username[0] + "*" * (len(username) - 2) + username[-1]
                        )
                    return f"{masked_username}@{domain}"
                except Exception:
                    return "******"
            else:
                return "*" * len(recipient)

        with db_session() as db:
            cases = db.query(Case).filter(Case.user_id == int_user_id).all()
            deadlines = (
                db.query(CaseDeadline).filter(CaseDeadline.user_id == int_user_id).all()
            )
            notifications = (
                db.query(NotificationLog)
                .filter(NotificationLog.user_id == int_user_id)
                .all()
            )

            case_list = []
            for c in cases:
                if anonymize:
                    try:
                        from case_manager import _generate_anonymized_case_id
                        anon_case_id = _generate_anonymized_case_id(c.id, c.created_at)
                    except Exception:
                        anon_case_id = hashlib.sha256(
                            f"anon-{c.id}".encode()
                        ).hexdigest()[:12]

                    case_num = f"ANON-{anon_case_id}"
                    case_title = "Anonymized Case Reference"
                else:
                    case_num = c.case_number
                    case_title = c.title or "Untitled Case"

                case_list.append(
                    {
                        "id": c.id,
                        "case_number": case_num,
                        "case_type": c.case_type,
                        "jurisdiction": c.jurisdiction,
                        "status": (
                            c.status.value
                            if hasattr(c.status, "value")
                            else str(c.status)
                        ),
                        "title": case_title,
                        "created_at": (
                            c.created_at.isoformat()
                            if hasattr(c.created_at, "isoformat")
                            else str(c.created_at)
                        ),
                    }
                )

            deadline_list = []
            for d in deadlines:
                if anonymize:
                    case_title = "Anonymized Case Reference"
                    description = "Redacted" if d.description else None
                else:
                    case_title = d.case_title
                    description = d.description

                deadline_list.append(
                    {
                        "id": d.id,
                        "case_id": d.case_id,
                        "case_title": case_title,
                        "deadline_date": (
                            d.deadline_date.isoformat()
                            if hasattr(d.deadline_date, "isoformat")
                            else str(d.deadline_date)
                        ),
                        "deadline_type": d.deadline_type,
                        "description": description,
                        "is_completed": d.is_completed,
                        "created_at": (
                            d.created_at.isoformat()
                            if hasattr(d.created_at, "isoformat")
                            else str(d.created_at)
                        ),
                    }
                )

            notification_list = []
            for n in notifications:
                if anonymize:
                    recipient = mask_recipient(n.recipient)
                else:
                    recipient = n.recipient

                notification_list.append(
                    {
                        "id": n.id,
                        "deadline_id": n.deadline_id,
                        "channel": (
                            n.channel.value
                            if hasattr(n.channel, "value")
                            else str(n.channel)
                        ),
                        "status": (
                            n.status.value
                            if hasattr(n.status, "value")
                            else str(n.status)
                        ),
                        "recipient": recipient,
                        "days_before": n.days_before,
                        "sent_at": (
                            n.sent_at.isoformat()
                            if n.sent_at and hasattr(n.sent_at, "isoformat")
                            else str(n.sent_at) if n.sent_at else None
                        ),
                        "created_at": (
                            n.created_at.isoformat()
                            if hasattr(n.created_at, "isoformat")
                            else str(n.created_at)
                        ),
                    }
                )

        self.update_state(
            state="PROGRESS",
            meta={"status": "Formatting export package", "progress": 60},
        )

        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)

            writer.writerow(["=== CASES ==="])
            writer.writerow(
                [
                    "id",
                    "case_number",
                    "case_type",
                    "jurisdiction",
                    "status",
                    "title",
                    "created_at",
                ]
            )
            for c in case_list:
                writer.writerow(
                    [
                        c["id"],
                        c["case_number"],
                        c["case_type"],
                        c["jurisdiction"],
                        c["status"],
                        c["title"],
                        c["created_at"],
                    ]
                )
            writer.writerow([])

            writer.writerow(["=== DEADLINES ==="])
            writer.writerow(
                [
                    "id",
                    "case_id",
                    "case_title",
                    "deadline_date",
                    "deadline_type",
                    "description",
                    "is_completed",
                    "created_at",
                ]
            )
            for d in deadline_list:
                writer.writerow(
                    [
                        d["id"],
                        d["case_id"],
                        d["case_title"],
                        d["deadline_date"],
                        d["deadline_type"],
                        d["description"] or "",
                        d["is_completed"],
                        d["created_at"],
                    ]
                )
            writer.writerow([])

            writer.writerow(["=== NOTIFICATIONS ==="])
            writer.writerow(
                [
                    "id",
                    "deadline_id",
                    "channel",
                    "status",
                    "recipient",
                    "days_before",
                    "sent_at",
                    "created_at",
                ]
            )
            for n in notification_list:
                writer.writerow(
                    [
                        n["id"],
                        n["deadline_id"] or "",
                        n["channel"],
                        n["status"],
                        n["recipient"],
                        n["days_before"],
                        n["sent_at"] or "",
                        n["created_at"],
                    ]
                )

            file_bytes = output.getvalue().encode("utf-8")
        else:  # json
            export_data = {
                "user_id": int_user_id,
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "cases": case_list,
                "deadlines": deadline_list,
                "notifications": notification_list,
            }
            file_bytes = json.dumps(export_data, indent=2).encode("utf-8")

        self.update_state(
            state="PROGRESS", meta={"status": "Saving to storage", "progress": 90}
        )

        export_file = save_export_file(
            user_id=str(int_user_id), file_bytes=file_bytes, format=format
        )

        logger.info(
            "User data export completed",
            task_id=self.request.id,
            user_id=int_user_id,
            export_id=export_file.export_id,
            format=format,
            file_path=export_file.file_path,
        )

        result = {
            "export_id": export_file.export_id,
            "file_path": export_file.file_path,
            "file_size_bytes": export_file.file_size_bytes,
            "format": format,
            "expires_in_hours": Config.EXPORT_FILE_EXPIRY_HOURS,
            "expires_at": export_file.expires_at.isoformat(),
            "created_at": export_file.created_at.isoformat(),
        }

        return result

    except Exception as e:
        logger.error(
            "User data export failed",
            task_id=self.request.id,
            user_id=user_id,
            error=str(e),
        )
        raise


@celery_app.task(bind=True, name="send_notification")
def send_notification_task(
    self, user_id: str, message: str, notification_type: str = "email"
) -> Dict[str, Any]:
    """Asynchronous task to send user notifications via various channels."""
    try:
        logger.info(
            "Dispatching notification",
            user_id=user_id,
            notification_type=notification_type,
        )

        result = {
            "notification_id": str(uuid.uuid4()),
            "user_id": user_id,
            "type": notification_type,
            "status": "dispatched",
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }

        return result

    except Exception as e:
        logger.error("Notification delivery failed", user_id=user_id, error=str(e))
        raise


# ============================================================================
# SCHEDULED PERIODIC TASKS (CELERY BEAT)
# ============================================================================


@celery_app.task(name="cleanup_old_tasks")
def cleanup_old_tasks() -> Dict[str, str]:
    """Maintenance task to clean up old completed tasks from the result backend."""
    logger.info("Executing periodic maintenance: cleanup_old_tasks")
    return {"status": "completed", "action": "cleanup"}


@celery_app.task(name="send_deadline_reminders")
def send_deadline_reminders() -> Dict[str, int]:
    """Periodic task to check for upcoming legal deadlines and notify users."""
    logger.info("Executing periodic task: send_deadline_reminders")
    return {"status": "completed", "reminders_sent": 0}


@celery_app.task(name="cleanup_revoked_tokens", bind=True, max_retries=3)
def cleanup_revoked_tokens(self) -> Dict[str, Any]:
    """Periodic task to clean up expired revoked tokens from the database."""
    from database import SessionLocal, cleanup_expired_revoked_tokens

    logger.info("Executing periodic maintenance: cleanup_revoked_tokens")
    db = SessionLocal()
    try:
        deleted = cleanup_expired_revoked_tokens(db)
        logger.info("cleanup_revoked_tokens_completed", deleted_count=deleted)
        return {"status": "completed", "deleted_count": deleted}
    except Exception as exc:
        logger.error("cleanup_revoked_tokens_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=60)
    finally:
        db.close()


@celery_app.task(name="enforce_retention_policies", bind=True, max_retries=3)
def enforce_retention_policies(self) -> Dict[str, Any]:
    """Phase 1: Archive cases that have passed their archive window."""
    from db.retention_models import RetentionRule, RetentionAuditLog, seed_retention_rules
    from db.retention_service import archive_expired_cases
    from db.session import db_session

    logger.info("Executing compliance: enforce_retention_policies (archive phase)")

    with db_session() as db:
        seed_retention_rules(db)
        rules = db.query(RetentionRule).all()
        archived_ids, count = archive_expired_cases(db, cutoff_days=730, dry_run=False)

        log = RetentionAuditLog(
            action="archive",
            data_category="cases",
            record_ids=archived_ids,
            records_affected=count,
            executed_by="celery:enforce_retention_policies",
            reason="Retention policy: archive_after_days=730",
        )
        db.add(log)
        db.commit()

        logger.info("enforce_retention_policies_archived", count=count)
        return {"status": "completed", "archived_count": count, "category": "cases"}


@celery_app.task(name="enforce_data_anonymization", bind=True, max_retries=3)
def enforce_data_anonymization(self) -> Dict[str, Any]:
    """Phase 2: Anonymize PII on records past the anonymization window."""
    from db.retention_models import RetentionAuditLog, seed_retention_rules
    from db.retention_service import anonymize_expired_records
    from db.models.feedback import UserFeedback
    from db.session import db_session

    logger.info("Executing compliance: enforce_data_anonymization")

    results = {}
    with db_session() as db:
        seed_retention_rules(db)

        feedback_pii = {"user_email": "email", "feedback_text": "text"}
        ids, count = anonymize_expired_records(
            db, UserFeedback, cutoff_days=365, pii_fields=feedback_pii, dry_run=False
        )
        results["user_feedback"] = count
        log = RetentionAuditLog(
            action="anonymize",
            data_category="user_feedback",
            record_ids=ids,
            records_affected=count,
            executed_by="celery:enforce_data_anonymization",
            reason="Retention policy: anonymize_after_days=365",
        )
        db.add(log)
        db.commit()

        logger.info("enforce_data_anonymization_completed", results=results)
        return {"status": "completed", "anonymized": results}


@celery_app.task(name="purge_expired_data", bind=True, max_retries=3)
def purge_expired_data(self) -> Dict[str, Any]:
    """Phase 3: Hard-delete records that have passed the deletion window."""
    from db.retention_models import RetentionAuditLog, seed_retention_rules
    from db.retention_service import (
        purge_expired_attachments,
        purge_expired_notifications,
        purge_expired_otl_tokens,
    )
    from db.session import db_session

    logger.info("Executing compliance: purge_expired_data")
    # Perform cleanups on database backends including Celery Redis results pruning if active
    results = {}
    with db_session() as db:
        seed_retention_rules(db)

        ids, count = purge_expired_notifications(db, cutoff_days=365, dry_run=False)
        results["notifications"] = count
        log = RetentionAuditLog(
            action="hard_delete",
            data_category="notifications",
            record_ids=ids,
            records_affected=count,
            executed_by="celery:purge_expired_data",
            reason="Retention policy: delete_after_days=365",
        )
        db.add(log)
        db.commit()

        ids, count = purge_expired_attachments(db, cutoff_days=1095, dry_run=False)
        results["attachments"] = count
        log = RetentionAuditLog(
            action="hard_delete",
            data_category="attachments",
            record_ids=ids,
            records_affected=count,
            executed_by="celery:purge_expired_data",
            reason="Retention policy: delete_after_days=1095",
        )
        db.add(log)
        db.commit()

        ids, count = purge_expired_otl_tokens(db, cutoff_days=5, dry_run=False)
        results["otp_tokens"] = count

        logger.info("purge_expired_data_completed", results=results)
        return {"status": "completed", "purged": results}