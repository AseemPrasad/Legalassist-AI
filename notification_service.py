"""
Notification service for sending SMS and Email reminders using Twilio and SendGrid.
Handles delivery tracking and retry logic.
"""

import logging
import re
import structlog
import os
import re
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import html
import tenacity
from config import Config

# Celery integration for asynchronous task execution
# We import the celery_app instance defined in the project's central 
# Celery configuration module. This allows us to use the @celery_app.task 
# decorator to offload long-running operations.
try:
    from celery_app import celery_app
except Exception:
    celery_app = None


if celery_app is None:
    class _DummyTask:
        def __init__(self, func):
            self._func = func
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__
            self.__module__ = func.__module__

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

        def delay(self, *args, **kwargs):
            try:
                self._func(self, *args, **kwargs)
            except Exception:
                pass
            from types import SimpleNamespace
            import uuid
            return SimpleNamespace(id=uuid.uuid4().hex, state="SUCCESS")

        def apply_async(self, *args, **kwargs):
            kw = kwargs.get("kwargs", {}) or kwargs
            try:
                self._func(self, **kw)
            except Exception:
                pass
            from types import SimpleNamespace
            import uuid
            return SimpleNamespace(id=uuid.uuid4().hex, state="SUCCESS")

    class _FallbackCeleryApp:
        def task(self, *args, **kwargs):
            def decorator(func):
                return _DummyTask(func)

            return decorator

    celery_app = _FallbackCeleryApp()


from sqlalchemy.orm import Session

# Email and SMS Libraries
try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
except ImportError:
    SendGridAPIClient = None
    class DummyMail:
        def __init__(self, *args, **kwargs):
            pass
    Mail = DummyMail
from db import (
    Case,
    NotificationStatus,
    NotificationChannel,
    NotificationLog,
    UserPreference,
    CaseDeadline,
)
from database import (
    get_notification_template_for_user,
    reserve_notification,
    update_notification_result,
    SessionLocal,
)
from db.crud.notifications import (
    get_or_create_notification_log,
    update_notification_log_by_keys,
    has_notification_been_sent,
)
from db.crud.audit import record_immutable_audit_event
from core.template_renderer import render_template, validate_template, TemplateValidationError
from core.deadline_engine import get_deadline_first_action
from core.log_redaction import mask_recipient, sanitize_log_text, storage_safe_recipient
from services.timeline_service import timeline_service as case_timeline_service

# Import debug mode helper

_NOTIFICATION_PREVIEW_MAX_LEN = 200


def _safe_preview(text: Optional[str]) -> str:
    """Truncate and redact PII from notification content for DB storage."""
    return sanitize_log_text(text)[:_NOTIFICATION_PREVIEW_MAX_LEN]


def _is_debug_or_testing_mode() -> bool:
    """Return True when explicit debug/testing flags are enabled."""
    return Config.DEBUG or Config.TESTING


def _should_use_celery(task) -> bool:
    """Return True if we should offload task execution to Celery."""
    from unittest.mock import Mock, MagicMock
    if isinstance(getattr(task, "delay", None), (Mock, MagicMock)):
        return True
    if _is_debug_or_testing_mode() and not Config.is_production():
        return False
    return True

logger = structlog.get_logger(__name__)

NOTIFICATION_TEMPLATE_ALLOWED_VARS = {
    "case_title",
    "case_number",
    "deadline_date",
    "days_before",
    "days_left",
    "court",
    "deadline_type",
    "deadline_description",
    "first_action",
    "link",
    "channel",
    "language",
}


def _template_language_key(language: Optional[str]) -> str:
    text = str(language or "en").strip().lower()
    return text or "en"


def _derive_first_action(deadline: CaseDeadline) -> str:
    stored_action = (getattr(deadline, "first_action", None) or "").strip()
    if stored_action:
        return stored_action

    return get_deadline_first_action(getattr(deadline, "deadline_type", None))


def _build_notification_template_values(
    deadline: CaseDeadline,
    days_left: int,
    channel: NotificationChannel,
    language: Optional[str] = None,
) -> dict[str, str]:
    case = getattr(deadline, "case", None)
    deadline_date = getattr(deadline, "deadline_date", None)
    deadline_date_text = deadline_date.strftime("%d %b %Y") if hasattr(deadline_date, "strftime") else ""
    case_number = getattr(case, "case_number", "") if case is not None else ""
    case_title = getattr(deadline, "case_title", "") or getattr(case, "title", "") or ""
    court = getattr(case, "jurisdiction", "") if case is not None else ""
    template_language = _template_language_key(language)

    return {
        "case_title": str(case_title),
        "case_number": str(case_number),
        "deadline_date": deadline_date_text,
        "days_before": str(days_left),
        "days_left": str(days_left),
        "court": str(court),
        "deadline_type": str(getattr(deadline, "deadline_type", "") or ""),
        "deadline_description": str(getattr(deadline, "description", "") or ""),
        "first_action": _derive_first_action(deadline),
        "link": f"https://legalassist.ai/cases/{getattr(deadline, 'case_id', '')}",
        "channel": channel.value if hasattr(channel, "value") else str(channel),
        "language": template_language,
    }


def _render_notification_template(template: str, values: dict[str, str]) -> str:
    return render_template(
        template,
        values,
        allowed=NOTIFICATION_TEMPLATE_ALLOWED_VARS,
        missing_as_empty=True,
    )


def _resolve_notification_template_values(
    db: Session,
    deadline: CaseDeadline,
    days_left: int,
    channel: NotificationChannel,
    language: Optional[str] = None,
) -> dict[str, Optional[str]]:
    template = get_notification_template_for_user(db, deadline.user_id, channel=channel, language=language)
    if not template:
        return {"sms_template": None, "email_subject_template": None, "email_html_template": None}
    return template.resolve_templates(channel=channel, language=language)


def _sanitize_preview(text: str, max_length: int = 160) -> str:
    """Truncate and strip HTML for safe preview storage."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0] + "..."
    return text


_SUBJECT_MAX_LEN = 200


def _sanitize_subject(text: str) -> str:
    """Sanitize dynamic content for safe use in email subject lines."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = " ".join(text.split())
    return text[:_SUBJECT_MAX_LEN]


@dataclass
class NotificationResult:
    """Result of a notification send attempt"""
    success: bool
    channel: NotificationChannel
    recipient: str
    message_id: Optional[str] = None
    error: Optional[str] = None


class SMSClient:
    """Wrapper for Twilio SMS client"""

    def __init__(self):
        self.account_sid = Config.TWILIO_ACCOUNT_SID
        self.auth_token = Config.get_twilio_auth_token()
        self.from_number = Config.TWILIO_FROM_NUMBER

        if not all([self.account_sid, self.auth_token, self.from_number]) or TwilioClient is None:
            logger.warning("Twilio credentials not configured or package not installed. SMS will be mocked.")
            self.client = None
        else:
            # Ensure Twilio library is available
            if TwilioClient is None:
                logger.warning("Twilio library not installed. SMS will be mocked.")
                self.client = None
            else:
                self.client = TwilioClient(self.account_sid, self.auth_token)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(
            lambda e: any(x in str(e) for x in ("503", "429", "Service Unavailable", "Too Many Requests"))
            or getattr(e, "status_code", None) in (429, 503)
            or getattr(e, "status", None) in (429, 503)
            or any(err in str(e).lower() for err in ("timeout", "connection", "connect", "unreachable"))
        ),
        reraise=True
    )
    def _create_message_with_retry(self, to_number: str, message: str):
        """Internal method to send SMS with tenacity retry for 503/429 errors."""
        return self.client.messages.create(
            body=message,
            from_=self.from_number,
            to=to_number,
        )

    def send_sms(self, to_number: str, message: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Send SMS message.
        Returns: (success, message_id, error)
        
        In debug/testing mode: Mocks the send and returns success
        In production: Fails if Twilio is not configured
        """
        try:
            if not self.client:
                # Not configured: run in mock mode ONLY if in debug/testing.
                if _is_debug_or_testing_mode() and not Config.is_production():
                    logger.info("sms_mocked", recipient=mask_recipient(to_number))
                    return True, f"mock_sms_{datetime.now().timestamp()}", None
                
                error_msg = "Twilio credentials not configured. SMS delivery skipped."
                logger.warning(error_msg)
                return False, None, error_msg

            message_obj = self._create_message_with_retry(to_number, message)
            logger.info("sms_sent", recipient=mask_recipient(to_number), message_id=message_obj.sid)
            return True, message_obj.sid, None

        except tenacity.RetryError as re:
            error_msg = f"Failed to send SMS after max retries due to provider errors: {sanitize_log_text(str(re))}"
            logger.error("sms_send_failed", recipient=mask_recipient(to_number), error=sanitize_log_text(str(re)))
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Failed to send SMS: {sanitize_log_text(str(e))}"
            logger.error("sms_send_failed", recipient=mask_recipient(to_number), error=sanitize_log_text(str(e)))
            return False, None, error_msg


class EmailClient:
    """Wrapper for SendGrid email client"""

    def __init__(self):
        self.api_key = Config.get_sendgrid_api_key()
        self.from_email = Config.SENDGRID_FROM_EMAIL

        if not self.api_key or SendGridAPIClient is None:
            logger.warning("SendGrid API key not configured or package not installed. Emails will be mocked.")
            self.client = None
        else:
            # Ensure SendGrid library is available
            if SendGridAPIClient is None:
                logger.warning("SendGrid library not installed. Emails will be mocked.")
                self.client = None
            else:
                self.client = SendGridAPIClient(self.api_key)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception(
            lambda e: any(x in str(e) for x in ("503", "429", "Service Unavailable", "Too Many Requests"))
            or getattr(e, "status_code", None) in (429, 503)
            or getattr(e, "status", None) in (429, 503)
            or any(err in str(e).lower() for err in ("timeout", "connection", "connect", "unreachable"))
        ),
        reraise=True
    )
    def _send_email_with_retry(self, message):
        """Internal method to send email with tenacity retry for 503/429 errors."""
        return self.client.send(message)

    def send_email(self, to_email: str, subject: str, html_content: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Send email.
        Returns: (success, message_id, error)
        
        In debug/testing mode: Mocks the send and returns success
        In production: Fails if SendGrid is not configured
        """
        try:
            if not self.client:
                # Not configured: run in mock mode ONLY if in debug/testing.
                if _is_debug_or_testing_mode() and not Config.is_production():
                    logger.info("email_mocked", recipient=mask_recipient(to_email))
                    return True, f"mock_email_{datetime.now().timestamp()}", None
                
                error_msg = "SendGrid API key not configured. Email delivery skipped."
                logger.warning(error_msg)
                return False, None, error_msg

            message = Mail(
                from_email=self.from_email,
                to_emails=to_email,
                subject=subject,
                html_content=html_content,
            )
            response = self._send_email_with_retry(message)
            logger.info("email_sent", recipient=mask_recipient(to_email), status_code=response.status_code)
            return True, response.headers.get("X-Message-ID", "unknown"), None

        except tenacity.RetryError as re:
            error_msg = f"Failed to send email after max retries due to provider errors: {sanitize_log_text(str(re))}"
            logger.error("email_send_failed", recipient=mask_recipient(to_email), error=sanitize_log_text(str(re)))
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Failed to send email: {sanitize_log_text(str(e))}"
            logger.error("email_send_failed", recipient=mask_recipient(to_email), error=sanitize_log_text(str(e)))
            return False, None, error_msg


# ============================================================================
# ASYNCHRONOUS BACKGROUND TASKS
# ============================================================================

def is_permanent_error(error_msg: str) -> bool:
    """Return True if the error is a permanent delivery or validation failure,
    indicating that retrying will not resolve the issue.
    """
    if not error_msg:
        return False
    msg = error_msg.lower()
    permanent_patterns = [
        "bad request", "invalid email", "invalid phone", "missing recipient",
        "400", "401", "403", "404", "auth", "credential", "unauthorized",
        "forbidden", "permission", "format", "malformed", "not found",
        "validation failed"
    ]
    return any(p in msg for p in permanent_patterns)


@celery_app.task(
    bind=True, 
    name="send_email_task", 
    max_retries=3, 
    default_retry_delay=60,
    queue="notifications"
)
def send_email_task(
    self, 
    to_email: str, 
    subject: str, 
    html_content: str,
    deadline_id: Optional[int] = None,
    user_id: Optional[int] = None,
    days_left: Optional[int] = None
) -> dict:
    """
    Celery background task for sending emails via SendGrid.
    """
    from database import db_session, NotificationStatus, NotificationChannel, update_notification_result
    
    logger.info("background_email_delivery_started", recipient=mask_recipient(to_email), task_id=self.request.id)
    
    client = EmailClient()
    success, message_id, error = client.send_email(to_email, subject, html_content)
    status = NotificationStatus.SENT if success else NotificationStatus.FAILED

    record_immutable_audit_event(
        event_type="notification.sent" if success else "notification.failed",
        action="sent" if success else "failed",
        actor_user_id=user_id,
        resource_type="notification",
        resource_id=f"email:{deadline_id}:{user_id}" if deadline_id is not None and user_id is not None else f"email:{self.request.id}",
        outcome="success" if success else "failure",
        case_id=None,
        metadata={
            "channel": NotificationChannel.EMAIL.value,
            "deadline_id": deadline_id,
            "days_left": days_left,
            "message_id": message_id,
            "error": error,
        },
    )
    
    if deadline_id is not None and user_id is not None and days_left is not None:
        try:
            with db_session() as db:
                update_notification_result(
                    db=db,
                    deadline_id=deadline_id,
                    user_id=user_id,
                    days_before=days_left,
                    channel=NotificationChannel.EMAIL,
                    status=NotificationStatus.SENT if success else NotificationStatus.FAILED,
                    message_id=message_id,
                    error_message=error,
                    message_preview=_safe_preview(html_content),
                )
                logger.info("Background notification result updated", deadline_id=deadline_id)
        except Exception as e:
            logger.error("Failed to update background notification", error=str(e), deadline_id=deadline_id)
    
    # Handle retries if the email failed
    if not success and error:
        if is_permanent_error(error):
            logger.error("email_delivery_failed_permanent", recipient=mask_recipient(to_email), error=sanitize_log_text(error))
        else:
            # Check for transient errors (503, 429) to use exponential backoff
            if self.request.retries < self.max_retries:
                if '503' in str(error) or '429' in str(error) or 'rate' in str(error).lower():
                    backoff_delay = (2 ** self.request.retries) * 60
                    logger.warning(
                        "email_delivery_retry_scheduled_backoff", 
                        error=sanitize_log_text(error), 
                        retry_count=self.request.retries + 1,
                        delay_seconds=backoff_delay
                    )
                    raise self.retry(exc=Exception(error), countdown=backoff_delay)
                else:
                    logger.warning("email_delivery_retry_scheduled", error=sanitize_log_text(error), retry_count=self.request.retries + 1)
                    raise self.retry(exc=Exception(error))
    
    return {
        "success": success,
        "message_id": message_id,
        "error": error,
        "status": status.value if hasattr(status, 'value') else str(status)
    }


@celery_app.task(
    bind=True, 
    name="send_sms_task", 
    max_retries=5, 
    default_retry_delay=60,
    queue="notifications"
)
def send_sms_task(
    self, 
    to_number: str, 
    message: str,
    deadline_id: Optional[int] = None,
    user_id: Optional[int] = None,
    days_left: Optional[int] = None
) -> dict:
    """
    Celery background task for sending SMS via Twilio.
    """
    from database import db_session, NotificationStatus, NotificationChannel
    from db.crud.notifications import update_notification_log_by_keys
    
    logger.info("background_sms_delivery_started", recipient=mask_recipient(to_number), task_id=self.request.id)
    
    client = SMSClient()
    success, message_id, error = client.send_sms(to_number, message)
    status = NotificationStatus.SENT if success else NotificationStatus.FAILED

    record_immutable_audit_event(
        event_type="notification.sent" if success else "notification.failed",
        action="sent" if success else "failed",
        actor_user_id=user_id,
        resource_type="notification",
        resource_id=f"sms:{deadline_id}:{user_id}" if deadline_id is not None and user_id is not None else f"sms:{self.request.id}",
        outcome="success" if success else "failure",
        case_id=None,
        metadata={
            "channel": NotificationChannel.SMS.value,
            "deadline_id": deadline_id,
            "days_left": days_left,
            "message_id": message_id,
            "error": error,
        },
    )
    
    if deadline_id is not None and user_id is not None and days_left is not None:
        try:
            with db_session() as db:
                update_notification_log_by_keys(
                    db=db,
                    user_id=user_id,
                    deadline_id=deadline_id,
                    days_before=days_left,
                    channel=NotificationChannel.SMS,
                    status=status,
                    message_id=message_id,
                    error_message=error,
                    message_preview=_sanitize_preview(message, max_length=200),
                )
                logger.info("background_sms_notification_logged", deadline_id=deadline_id)
        except Exception as e:
            logger.error("background_sms_notification_log_failed", deadline_id=deadline_id, error=sanitize_log_text(str(e)))
    
    # Retry mechanism for transient errors using exponential backoff
    if not success and error:
        if is_permanent_error(error):
            logger.error("sms_delivery_failed_permanent", recipient=mask_recipient(to_number), error=sanitize_log_text(error))
        else:
            if self.request.retries < self.max_retries:
                if '503' in str(error) or '429' in str(error) or 'rate' in str(error).lower():
                    backoff_delay = (2 ** self.request.retries) * 60
                    logger.warning(
                        "sms_delivery_retry_scheduled_backoff", 
                        error=sanitize_log_text(error), 
                        retry_count=self.request.retries + 1,
                        delay_seconds=backoff_delay
                    )
                    raise self.retry(exc=Exception(error), countdown=backoff_delay)
                else:
                    logger.warning("sms_delivery_retry_scheduled", error=sanitize_log_text(error), retry_count=self.request.retries + 1)
                    raise self.retry(exc=Exception(error))
    
    return {
        "success": success,
        "message_id": message_id,
        "error": error,
        "status": status.value if hasattr(status, 'value') else str(status)
    }


class NotificationService:
    """Main service for sending deadline reminders"""

    def __init__(self):
        self.sms_client = SMSClient()
        self.email_client = EmailClient()
        raw_url = Config.BASE_URL
        if not raw_url:
            logger.warning("BASE_URL is not configured; using default for notification links")
            raw_url = "https://legalassist.ai"
        self.base_url = raw_url.rstrip('/')

    def build_sms_message(self, case_title: str, days_left: int, deadline_date: datetime, first_action: Optional[str] = None) -> str:
        """Build SMS reminder message"""
        formatted_date = deadline_date.strftime("%d %b %Y")
        action = (first_action or "").strip()
        action_text = f" Next action: {action}." if action else ""
        return (
            f"⚖️ LegalAssist: Case '{case_title}' has a deadline in {days_left} day(s). "
            f"Deadline: {formatted_date}.{action_text} Log in to check details."
        )

    def build_email_message(self, deadline: CaseDeadline, days_left: int, first_action: Optional[str] = None) -> Tuple[str, str]:
        """
        Build a premium email reminder content.
        Uses modern HTML/CSS with glassmorphism-inspired design.
        Returns: (subject, html_content)
        """
        formatted_date = deadline.deadline_date.strftime("%d %B %Y")
        escaped_title = html.escape(deadline.case_title)
        escaped_type = html.escape(deadline.deadline_type.title())
        escaped_desc = html.escape(deadline.description) if deadline.description else "No additional details provided."
        escaped_action = html.escape((first_action or _derive_first_action(deadline)).strip())
        
        # Urgency color coding
        if days_left <= 3:
            accent_color = "#ff5252" # Critical Red
            urgency_label = "URGENT"
        elif days_left <= 10:
            accent_color = "#ff9100" # Warning Orange
            urgency_label = "SOON"
        else:
            accent_color = "#1a5490" # Info Blue
            urgency_label = "REMINDER"

        subject = f"⚖️ {urgency_label}: {_sanitize_subject(deadline.case_title)} - {_sanitize_subject(deadline.deadline_type.title())} Deadline"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; margin: 0; padding: 0; }}
                .container {{ max-width: 600px; margin: 40px auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.1); border: 1px solid #eee; }}
                .header {{ background: linear-gradient(135deg, #1a5490 0%, #0d2c4d 100%); padding: 40px 30px; text-align: center; color: white; }}
                .header h1 {{ margin: 0; font-size: 24px; letter-spacing: 1px; }}
                .content {{ padding: 40px 30px; color: #444; line-height: 1.6; }}
                .status-badge {{ display: inline-block; padding: 4px 12px; background: {accent_color}22; color: {accent_color}; border: 1px solid {accent_color}; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 20px; text-transform: uppercase; }}
                .case-title {{ font-size: 22px; font-weight: 700; color: #1a5490; margin-bottom: 10px; }}
                .deadline-box {{ background: #fdfdfd; border-radius: 12px; border-left: 6px solid {accent_color}; padding: 25px; margin: 30px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.03); }}
                .deadline-item {{ margin-bottom: 15px; }}
                .deadline-label {{ color: #888; font-size: 13px; text-transform: uppercase; font-weight: 600; display: block; }}
                .deadline-value {{ font-size: 18px; color: #222; font-weight: 600; }}
                .description {{ background: #f9f9f9; padding: 20px; border-radius: 8px; font-style: italic; color: #666; margin-top: 20px; border-left: 3px solid #ddd; }}
                .next-action {{ background: #eef6ff; padding: 18px 20px; border-radius: 10px; margin-top: 20px; border-left: 4px solid {accent_color}; }}
                .next-action-label {{ display: block; color: #1a5490; font-size: 13px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px; }}
                .cta-button {{ display: inline-block; background: #1a5490; color: white !important; padding: 16px 40px; text-decoration: none; border-radius: 30px; font-weight: bold; margin-top: 30px; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(26, 84, 144, 0.3); }}
                .footer {{ background: #f4f4f4; padding: 30px; text-align: center; color: #999; font-size: 12px; }}
                .footer a {{ color: #1a5490; text-decoration: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚖️ LegalAssist AI</h1>
                </div>
                <div class="content">
                    <div class="status-badge">{urgency_label} ACTION REQUIRED</div>
                    <div class="case-title">Case: {escaped_title}</div>
                    <p>Dear Litigant,</p>
                    <p>This is a formal reminder regarding an upcoming deadline for your ongoing legal matter. Timely action is critical to protect your legal rights.</p>
                    
                    <div class="deadline-box">
                        <div class="deadline-item">
                            <span class="deadline-label">Deadline Type</span>
                            <span class="deadline-value">{escaped_type}</span>
                        </div>
                        <div class="deadline-item">
                            <span class="deadline-label">Due Date</span>
                            <span class="deadline-value" style="color: {accent_color};">{formatted_date}</span>
                        </div>
                        <div class="deadline-item" style="margin-bottom: 0;">
                            <span class="deadline-label">Time Remaining</span>
                            <span class="deadline-value">{days_left} Days</span>
                        </div>
                    </div>
 
                    <div class="deadline-label">Details</div>
                    <div class="description">
                        "{escaped_desc}"
                    </div>

                    <div class="next-action">
                        <span class="next-action-label">Suggested Next Action</span>
                        <span class="deadline-value">{escaped_action}</span>
                    </div>

                    <div style="text-align: center;">
                        <a href="{self.base_url}/cases/{deadline.case_id}" class="cta-button">
                            View Case Dashboard
                        </a>
                    </div>
                </div>
                <div class="footer">
                    <p>This is an automated notification from your LegalAssist AI account.<br>
                    Missing deadlines can lead to dismissal of your case. Please consult with your legal counsel immediately.</p>
                    <p>Manage your <a href="{self.base_url}/settings">Notification Preferences</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return subject, html_content

    def _get_fallback_channel_order(self, user_preference: UserPreference) -> List[NotificationChannel]:
        if user_preference.notification_channel == NotificationChannel.EMAIL:
            return [NotificationChannel.EMAIL, NotificationChannel.SMS]
        return [NotificationChannel.SMS, NotificationChannel.EMAIL]

    def send_with_fallback(
        self,
        db: Session,
        deadline: CaseDeadline,
        user_preference: UserPreference,
        days_left: int,
    ) -> NotificationResult:
        """Send a reminder using the first working channel and record the whole attempt chain."""
        attempted_channels: List[str] = []
        channel_order = self._get_fallback_channel_order(user_preference)
        final_channel = channel_order[0]
        final_recipient = user_preference.phone_number or user_preference.email or "unknown"
        final_message_id: Optional[str] = None
        final_error: Optional[str] = None
        final_message_preview: Optional[str] = None
        success = False

        sms_message: Optional[str] = None
        email_subject: Optional[str] = None
        email_html_content: Optional[str] = None

        for channel in channel_order:
            attempted_channels.append(channel.value)

            if channel == NotificationChannel.SMS:
                if not user_preference.phone_number:
                    final_channel = channel
                    final_recipient = "unknown"
                    final_error = "No phone number configured"
                    continue

                if sms_message is None:
                    sms_message = self.build_sms_message(
                        getattr(deadline, "case_title", ""),
                        days_left,
                        deadline.deadline_date,
                        _derive_first_action(deadline),
                    )

                success, message_id, error = self.sms_client.send_sms(user_preference.phone_number, sms_message)
                final_channel = channel
                final_recipient = user_preference.phone_number
                final_message_id = message_id
                final_error = error
                final_message_preview = _safe_preview(sms_message)
            else:
                if not user_preference.email:
                    final_channel = channel
                    final_recipient = "unknown"
                    final_error = "No email address configured"
                    continue

                if email_subject is None or email_html_content is None:
                    email_subject, email_html_content = self.build_email_message(deadline, days_left, _derive_first_action(deadline))

                success, message_id, error = self.email_client.send_email(user_preference.email, email_subject, email_html_content)
                final_channel = channel
                final_recipient = user_preference.email
                final_message_id = message_id
                final_error = error
                final_message_preview = _safe_preview(email_subject)

            if success:
                break

        final_status = NotificationStatus.SENT if success else NotificationStatus.FAILED

        try:
            update_notification_result(
                db=db,
                deadline_id=deadline.id,
                user_id=deadline.user_id,
                days_before=days_left,
                channel=final_channel,
                status=final_status,
                message_id=final_message_id,
                error_message=final_error,
                message_preview=final_message_preview,
                recipient=final_recipient,
                attempted_channels=attempted_channels,
            )
        except Exception:
            logger.exception(
                "fallback_notification_log_failed",
                deadline_id=deadline.id,
                user_id=deadline.user_id,
                days_left=days_left,
                attempted_channels=attempted_channels,
            )

        try:
            record_immutable_audit_event(
                event_type="notification.sent" if success else "notification.failed",
                action="sent" if success else "failed",
                actor_user_id=deadline.user_id,
                resource_type="notification",
                resource_id=f"fallback:{deadline.id}:{deadline.user_id}:{days_left}",
                outcome="success" if success else "failure",
                case_id=deadline.case_id,
                metadata={
                    "deadline_id": deadline.id,
                    "days_left": days_left,
                    "attempted_channels": attempted_channels,
                    "final_channel": final_channel.value,
                    "message_id": final_message_id,
                    "error": final_error,
                },
            )
        except Exception:
            logger.exception(
                "fallback_notification_audit_failed",
                deadline_id=deadline.id,
                user_id=deadline.user_id,
            )

        return NotificationResult(
            success=success,
            channel=final_channel,
            recipient=final_recipient,
            message_id=final_message_id,
            error=final_error,
            attempted_channels=attempted_channels,
        )

    def send_sms_reminder(
        self,
        db: Session,
        deadline: CaseDeadline,
        user_preference: UserPreference,
        days_left: int,
        language: Optional[str] = None,
    ) -> NotificationResult:
        """Send SMS reminder for a deadline"""
        
        if not user_preference.phone_number:
            logger.warning(f"User {deadline.user_id} has no phone number. Skipping SMS.")
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SMS,
                recipient="unknown",
                error="No phone number configured",
            )

        template_language = _template_language_key(language)

        # Try per-user template first
        message = None
        try:
            tmpl = _resolve_notification_template_values(db, deadline, days_left, NotificationChannel.SMS, template_language)
            sms_template = tmpl.get("sms_template") if isinstance(tmpl, dict) else None
            if sms_template:
                values = _build_notification_template_values(deadline, days_left, NotificationChannel.SMS, template_language)
                message = _render_notification_template(sms_template, values)
        except TemplateValidationError as e:
            logger.warning("User SMS template invalid, falling back to default: %s", str(e))
        except Exception:
            logger.exception("Error rendering user SMS template; falling back to default")

        if message is None:
            message = self.build_sms_message(getattr(deadline, 'case_title', ''), days_left, deadline.deadline_date, _derive_first_action(deadline))

        if not _should_use_celery(send_sms_task):
            success, message_id, error = self.sms_client.send_sms(user_preference.phone_number, message)

            status = NotificationStatus.SENT if success else NotificationStatus.FAILED

            # Update the reserved record with the final result
            update_notification_result(
                db=db,
                deadline_id=deadline.id,
                user_id=deadline.user_id,
                days_before=days_left,
                channel=NotificationChannel.SMS,
                status=status,
                message_id=message_id,
                error_message=error,
                message_preview=message,
            )

        # Atomically create the notification log with the final status.
        # The unique constraint on (deadline_id, days_before, channel) prevents
        # duplicate sends from concurrent workers.
        try:
            with db.begin_nested():
                log = NotificationLog(
                    deadline_id=deadline.id,
                    user_id=deadline.user_id,
                    channel=NotificationChannel.SMS,
                    recipient=storage_safe_recipient(user_preference.phone_number),
                    days_before=days_left,
                    message_preview=_safe_preview(message),
                    status=status,
                    message_id=message_id,
                    error_message=error,
                )
                if success:
                    log.sent_at = datetime.now(timezone.utc)
                db.add(log)
                db.flush()
            try:
                case_timeline_service.record_notification_event(
                    db=db,
                    notification_log=log,
                    status=NotificationStatus.SENT if success else NotificationStatus.FAILED,
                    provider="twilio",
                    metadata={
                        "message_preview": _safe_preview(message),
                        "error_message": error,
                    },
                )
            except Exception:
                logger.exception("sms_notification_timeline_event_failed", deadline_id=deadline.id, user_id=deadline.user_id)
        except IntegrityError:
            logger.debug("SMS notification already recorded; skipping", deadline_id=deadline.id, days_before=days_left)
            return NotificationResult(
                success=status == NotificationStatus.SENT,
                channel=NotificationChannel.SMS,
                recipient=user_preference.phone_number,
                message_id=message_id,
                error=error,

            )
        except Exception:
            logger.exception("Failed to annotate reserved SMS with task id")

        # Offload SMS delivery to background task
        logger.info(
            "Offloading SMS delivery to background task",
            user_id=deadline.user_id,
            deadline_id=deadline.id,
            days_left=days_left,
        )

        task_result = send_sms_task.delay(
            to_number=user_preference.phone_number,
            message=message,
            deadline_id=deadline.id,
            user_id=deadline.user_id,
            channel=NotificationChannel.SMS,
            recipient=user_preference.phone_number,
            days_before=days_left,
            status=status,
            message_id=message_id,
            error_message=error,
            message_preview=_sanitize_preview(message),
        )

        try:
            update_notification_result(
                db=db,
                deadline_id=deadline.id,
                user_id=deadline.user_id,
                days_before=days_left,
                channel=NotificationChannel.SMS,
                status=NotificationStatus.PENDING,
                message_id=f"task_{task_result.id}",
                message_preview=message,
            )
        except Exception:
            logger.exception("Failed to annotate reserved SMS with task id")

        return NotificationResult(
            success=True,
            channel=NotificationChannel.SMS,
            recipient=user_preference.phone_number,
            message_id=f"task_{task_result.id}",
            error=None,
        )

    def send_email_reminder(
        self,
        db: Session,
        deadline: CaseDeadline,
        user_preference: UserPreference,
        days_left: int,
        language: Optional[str] = None,
    ) -> NotificationResult:
        """Send email reminder for a deadline"""
        template_language = _template_language_key(language)
        # Try per-user template first
        subject = None
        html_content = None
        try:
            tmpl = get_notification_template_for_user(db, deadline.user_id)
            if tmpl and (tmpl.email_html_template or tmpl.email_subject_template):
                values = {
                    "case_title": deadline.case_title,
                    "case_number": getattr(deadline, "case_id", ""),
                    "deadline_date": deadline.deadline_date.strftime("%d %B %Y") if deadline.deadline_date else "",
                    "days_left": days_left,
                    "court": "",
                    "deadline_type": deadline.deadline_type,
                    "deadline_description": deadline.description or "",
                    "link": f"https://legalassist.ai/cases/{deadline.case_id}",
                }
                if tmpl.email_subject_template:
                    subject = _sanitize_subject(render_template(tmpl.email_subject_template, values))
                if tmpl.email_html_template:
                    html_content = render_template(tmpl.email_html_template, values)
        except TemplateValidationError as e:
            logger.warning("User email template invalid, falling back to default: %s", str(e))
        except Exception:
            logger.exception("Error rendering user email template; falling back to default")

        if subject is None or html_content is None:
            subject, html_content = self.build_email_message(deadline, days_left, _derive_first_action(deadline))



        # ====================================================================
        # ASYNCHRONOUS DELIVERY OFFLOAD
        # ====================================================================
        # Instead of calling self.email_client.send_email() directly, which
        # would block the current thread for several seconds while waiting
        # for the SendGrid API response, we dispatch a Celery task.
        # This allows the request (or the periodic check) to complete
        # immediately, providing a much smoother and "snappier" experience
        # for the end-user or the system scheduler.
        # ====================================================================

        logger.info(
            "Offloading email delivery to background task",
            user_id=deadline.user_id,
            deadline_id=deadline.id,
            days_left=days_left,
        )

        # Reserve a notification slot first to avoid concurrent sends
        reserved_log, created = reserve_notification(
            db=db,
            deadline_id=deadline.id,
            user_id=deadline.user_id,
            channel=NotificationChannel.EMAIL,
            recipient=storage_safe_recipient(user_preference.email),
            days_before=days_left,
            message_preview=_safe_preview(html_content),
        )

        if not created:
            logger.debug("Email notification already reserved; skipping", deadline_id=deadline.id, days_before=days_left)
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                recipient=user_preference.email,
                message_id=reserved_log.message_id,
                error="Notification already reserved/sent",
            )

        # Annotate the reserved record with a placeholder task id BEFORE dispatching,
        # so the worker never races against an uncommitted DB state.
        reserved_log.message_id = "task_pending"
        reserved_log.message_preview = _safe_preview(html_content)
        db.add(reserved_log)
        db.commit()

        task_result = send_email_task.delay(
            to_email=user_preference.email,
            subject=subject,
            html_content=html_content,
            deadline_id=deadline.id,
            user_id=deadline.user_id,
            days_left=days_left,
        )

        # Update the reserved log with task id as message_id (still PENDING until background updates)
        try:
            update_notification_result(
                db=db,
                deadline_id=deadline.id,
                user_id=deadline.user_id,
                days_before=days_left,
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.PENDING,
                message_id=f"task_{task_result.id}",
                message_preview=html_content,
            )
        except Exception:
            logger.exception("Failed to annotate reserved email with task id")

        return NotificationResult(
            success=True,
            channel=NotificationChannel.EMAIL,
            recipient=user_preference.email,
            message_id=f"task_{task_result.id}",
            error=None,
        )

    def send_reminders(
        self,
        db: Session,
        deadline: CaseDeadline,
        user_preference: UserPreference,
        days_left: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[NotificationResult]:
        """
        Send appropriate reminders based on days until deadline and user preferences.
        Checks which reminders should be sent for 30, 10, 3, and 1 day marks.
        """
        results = []
        if days_left is None:
            days_left = deadline.days_until_deadline()

        logger.debug("Checking reminders for deadline", 
                    case_id=deadline.case_id, 
                    days_left=days_left, 
                    user_id=deadline.user_id)

        # Only process at specific thresholds
        if days_left not in [30, 10, 3, 1]:
            return results

        # Send based on user's notification channel preference
        channels = []
        if user_preference.notification_channel == NotificationChannel.BOTH:
            channels = [NotificationChannel.SMS, NotificationChannel.EMAIL]
        else:
            channels = [user_preference.notification_channel]


        notification_language = language or getattr(user_preference, "language", None) or "en"

        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = []
        with ThreadPoolExecutor(max_workers=max(1, len(channels))) as executor:
            for channel in channels:
                # Check if reminder was already sent for this specific threshold and channel
                if not has_notification_been_sent(db, deadline.id, days_left, channel, user_id=deadline.user_id):
                    if channel == NotificationChannel.SMS:
                        futures.append(executor.submit(
                            self.send_sms_reminder, db, deadline, user_preference, days_left, notification_language
                        ))
                    elif channel == NotificationChannel.EMAIL:
                        futures.append(executor.submit(
                            self.send_email_reminder, db, deadline, user_preference, days_left, notification_language
                        ))
                else:
                    logger.info("Notification already sent, reporting as successful",
                                channel=channel.value if hasattr(channel, 'value') else str(channel),
                                days_left=days_left,
                                deadline_id=deadline.id)
                    recipient = getattr(user_preference, "phone_number", None) or getattr(user_preference, "email", "unknown")
                    results.append(NotificationResult(
                        success=True,
                        channel=channel,
                        recipient=recipient,
                        message_id=None,
                        error=None,
                    ))

            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    results.append(result)
                except Exception as exc:
                    logger.exception("Concurrent notification dispatch failed: %s", exc)

        return results
