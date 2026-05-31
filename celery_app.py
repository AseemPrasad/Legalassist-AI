import os
import uuid
import structlog
import json
import re
import hashlib
import io
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from types import SimpleNamespace
from celery import Celery, Task
from celery.result import AsyncResult
from api.validation import validate_file_url, fetch_url_safe

# Database & Core Imports
from database import SessionLocal, DocumentProcessingState
from core.app_utils import (
    PipelineStateManager, extract_text_from_pdf, get_client, 
    build_prompt, build_remedies_prompt, parse_remedies_response, 
    compress_text
)
from api.idempotency import IdempotencyManager
from api.validation import ValidationConfig
from db.crud.reports import update_report_status
from db.session import db_session
from database import Attachment, User, SessionLocal, get_case_by_id, get_case_document_by_id, update_case_document, create_timeline_event

# ============================================================================
# INITIALIZATION & LOGGING
# ============================================================================

# Initialize the settings object to fetch global configurations
settings = get_settings()

# Initialize
logger = structlog.get_logger(__name__)

# Ensure Celery instrumentation and HTTP instrumentation are active in workers
try:
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    try:
        CeleryInstrumentor().instrument()
    except Exception:
        pass
    try:
        RequestsInstrumentor().instrument()
    except Exception:
        pass
except Exception:
    # If opentelemetry not available, skip instrumentation silently
    pass


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
    # Dummy stub so @celery_app.task decorators and .conf.update() don't crash
    celery_app = SimpleNamespace()
    celery_app.conf = SimpleNamespace()
    celery_app.conf.update = lambda **kw: None
    celery_app.conf.__setitem__ = lambda k, v: None
    celery_app.Task = lambda: None
    def _stub_task(*dec_args, **dec_kwargs):
        if dec_args and callable(dec_args[0]):
            return _stub_task()(dec_args[0])
        def _wrapper(f):
            if dec_kwargs.get("bind"):
                def _bound(*args, **kwargs):
                    dummy_self = SimpleNamespace(
                        request=SimpleNamespace(id=str(uuid.uuid4()), retries=0),
                        retry=lambda *a, **kw: None,
                        update_state=lambda *a, **kw: None,
                    )
                    return f(dummy_self, *args, **kwargs)
                _bound.__name__ = f.__name__
                _bound.__qualname__ = f.__qualname__
                return _bound
            return f
        return _wrapper
    celery_app.task = _stub_task
    celery_app.AsyncResult = lambda *args, **kwargs: SimpleNamespace(state="PENDING", result=None, status="PENDING")
    celery_app.main = "legalassist"
    REDIS_URL = ""
else:
    REDIS_URL = _redis_env

    # Initialize the Celery application instance
    celery_app = Celery(
        "legalassist", broker=REDIS_URL, backend=REDIS_URL, task_cls=ContextTask
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
    task_time_limit=get_settings().CELERY_TASK_TIMEOUT,
    task_soft_time_limit=get_settings().CELERY_TASK_SOFT_TIME_LIMIT,
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
            info = (
                result.info
                if isinstance(result.info, dict)
                else {"status": "Processing"}
            )

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
def analyze_document_task(self, user_id, document_id, text=None, file_bytes=None, document_type="unknown", file_path=None, file_url=None) -> Dict[str, Any]:
    db = SessionLocal()
    idemp = IdempotencyManager()
    idempotency_key = f"doc:{user_id}:{document_id}"
    start_time = datetime.now(timezone.utc)
    
    # 1. State Recovery
    state = PipelineStateManager.get_state(db, document_id)
    stage = state.current_stage if state else "PENDING"
    data = state.stage_data if state else {}

    try:
        # Phase 1: Text Pre-processing
        self.update_state(
            state="PROGRESS",
            meta={"status": "Extracting and cleaning text", "progress": 25},
        )
        idemp.heartbeat(idempotency_key, ttl=300)

        logger.info(
            "Starting document analysis",
            task_id=self.request.id,
            user_id=user_id,
            document_id=document_id,
        )
        
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
                pinned = validate_file_url(file_url)
                resp = fetch_url_safe(pinned, timeout=30)
                if len(resp.content) > ValidationConfig.MAX_TEXT_LENGTH:
                    raise ValueError(f"Downloaded file too large: {len(resp.content)} bytes exceeds limit of {ValidationConfig.MAX_TEXT_LENGTH} bytes.")
                content_type = resp.headers.get("Content-Type", "")
                if "application/pdf" in content_type or file_url.lower().endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(io.BytesIO(resp.content))
                else:
                    extracted_text = resp.content.decode("utf-8", errors="ignore")
            elif file_path:
                # Ownership verification: the user must own an Attachment for this path
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

        # Enforce size limits
        if len(extracted_text.encode("utf-8")) > ValidationConfig.MAX_TEXT_LENGTH:
            raise ValueError(f"Extracted text exceeds max limit of {ValidationConfig.MAX_TEXT_LENGTH} bytes.")

        # Phase 2: Content Analysis
        self.update_state(
            state="PROGRESS", meta={"status": "Analyzing legal content", "progress": 50}
        )
        idemp.heartbeat(idempotency_key, ttl=300)
        
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
        # Extract JSON bullets if possible, otherwise use raw text
        summary_text = ""
        key_points = []
        try:
            import json
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_summary, re.DOTALL)
            json_str = match.group(1) if match else raw_summary
            data = json.loads(json_str)
            key_points = data.get("bullets", [])
            summary_text = " ".join(key_points)
        except Exception:
            summary_text = raw_summary

        # Phase 3: Remedy Extraction
        self.update_state(
            state="PROGRESS",
            meta={"status": "Extracting identified remedies", "progress": 75},
        )
        idemp.heartbeat(idempotency_key, ttl=300)
        
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

        # Phase 4: Finalization
        self.update_state(
            state="PROGRESS",
            meta={"status": "Finalizing analysis results", "progress": 90},
        )
        idemp.heartbeat(idempotency_key, ttl=300)
        
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Combine remedies into a structured array
        remedies_list = []
        if remedies_data.get("first_action"):
            remedies_list.append(f"Action: {remedies_data['first_action']}")
        if remedies_data.get("can_appeal") == "yes":
            remedies_list.append(f"Appeal allowed in {remedies_data.get('appeal_court', 'court')} within {remedies_data.get('appeal_days', 'unknown')} days.")
            
        deadlines_list = []
        if remedies_data.get("deadline"):
            deadlines_list.append(remedies_data["deadline"])

        result = {
            "document_id": document_id,
            "title": "Analyzed Document",
            "document_type": document_type,
            "summary": summary_text,
            "key_points": key_points,
            "remedies": remedies_list,
            "deadlines": deadlines_list,
            "obligations": [],
            "confidence_score": 0.85 if not remedies_data.get("_is_partial") else 0.6,
            "remedies_confidence_score": remedies_data.get("confidence_score", 0.0),
            "remedies_evidence_spans": remedies_data.get("evidence_spans", []),
            "analysis_time_seconds": analysis_time,
            "processed_at": datetime.now(timezone.utc).isoformat()

        }

        logger.info(
            "Document analysis completed",
            task_id=self.request.id,
            document_id=document_id,
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
            error=str(e),
        )
        # Re-raise the exception to trigger Celery's retry mechanism
        raise
    finally:
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
    """
    Asynchronous task to generate a formal report for a legal case.

    Args:
        user_id (str): The ID of the user requesting the report.
        case_id (str): The ID of the case for which the report is generated.
        report_id (str): Unique report UUID created by API.
        report_type (str): The type of report (e.g., 'summary', 'comprehensive').
        format (str): The output format ('pdf', 'html', etc.).
    Returns:
        Dict[str, Any]: Metadata about the generated report file.
    """
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
    idempotency_key = f"report:{user_id}:{case_id}:{report_type}:{format}:{privacy_profile}"
    if not idemp.acquire(key=idempotency_key, ttl=600):
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

    try:
        # Mark task as started in DB
        db = next(get_db())
        update_report_status(
            db,
            report_id,
            status="processing",
            started_at=datetime.utcnow()
        )
        
        # Step 1: Data Aggregation
        self.update_state(
            state="PROGRESS",
            meta={"status": "Compiling case data and documents", "progress": 20},
        )
        idemp.heartbeat(idempotency_key, ttl=600)

        logger.info(
            "Starting report generation",
            task_id=self.request.id,
            user_id=user_id,
            case_id=case_id,
        )

        # Step 2: Content Formatting
        self.update_state(
            state="PROGRESS",
            meta={"status": "Formatting document structure", "progress": 50},
        )
        idemp.heartbeat(idempotency_key, ttl=600)

        # Step 3: Rendering
        self.update_state(
            state="PROGRESS",
            meta={"status": "Rendering output document", "progress": 80},
        )
        idemp.heartbeat(idempotency_key, ttl=600)

        # Finalization
        self.update_state(
            state="PROGRESS",
            meta={"status": "Finalizing report generation", "progress": 95},
        )
        idemp.heartbeat(idempotency_key, ttl=600)

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
            privacy_profile=privacy_profile,
        )
        idemp.heartbeat(idempotency_key, ttl=600)

        # Update Report record with completion details
        file_path_str = str(generated.file_path)
        db = next(get_db())
        update_report_status(
            db,
            report_id,
            status="completed",
            file_path=file_path_str,
            file_size_bytes=generated.file_size_bytes,
            completed_at=datetime.utcnow()
        )

        # Prepare the result metadata for the frontend
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
        # Mark report as failed in DB
        try:
            db = next(get_db())
            update_report_status(
                db,
                report_id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.utcnow()
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

    Exports user data and saves to local storage with real file path.

    Args:
        user_id (str): The ID of the user whose data is being exported.
        format (str): The desired export format (csv, json). Default: csv
        anonymize (bool): Whether to anonymize sensitive user data (PII) before export.

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
            state="PROGRESS", meta={"status": "Gathering user data", "progress": 30}
        )

        # Validate format
        if format not in ("csv", "json"):
            return {
                "export_id": None,
                "file_path": None,
                "file_size_bytes": 0,
                "format": format,
                "expires_in_hours": None,
                "expires_at": None,
                "created_at": None,
            }

        try:
            int_user_id = int(user_id)
        except (ValueError, TypeError) as e:
            logger.error(
                "Invalid user ID format during export", user_id=user_id, error=str(e)
            )
            raise ValueError(f"Invalid user_id: {user_id}. Must be an integer.")

        import csv
        import io
        import hashlib
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
                digits = [c for c in recipient if c.isdigit()]
                if len(digits) >= 7:
                    return recipient[:3] + "*" * (len(recipient) - 7) + recipient[-4:]
                else:
                    return "*******"

        with db_session() as db:
            # Query real user records
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

        # Serialize based on format
        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)

            # --- CASES ---
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
            writer.writerow([])  # separator

            # --- DEADLINES ---
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
            writer.writerow([])  # separator

            # --- NOTIFICATIONS ---
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

        # Save to storage and get metadata
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
            notification_type=notification_type,
        )

        # Logic for sending notifications would go here
        # (e.g., integration with SendGrid, Twilio, or Firebase)

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


@celery_app.task(name="cleanup_revoked_tokens", bind=True, max_retries=3)
def cleanup_revoked_tokens(self) -> Dict[str, Any]:
    """
    Periodic task to clean up expired revoked tokens from the database.
    Prevents unbounded blacklist growth in distributed deployments.
    """
    from database import SessionLocal, cleanup_expired_revoked_tokens

    db = SessionLocal()
    try:
        deleted = cleanup_expired_revoked_tokens(db)
        return {"status": "completed", "deleted_tokens": deleted}
    except Exception as e:
        db.rollback()
        logger.error("cleanup_revoked_tokens_failed", error=str(e))
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()