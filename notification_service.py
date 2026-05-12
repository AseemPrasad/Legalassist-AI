"""
Notification service for sending SMS and Email reminders using Twilio and SendGrid.
Handles delivery tracking and retry logic.
"""

import logging
import structlog
import os
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import html
from config import Config

# Celery integration for asynchronous task execution
# We import the celery_app instance defined in the project's central 
# Celery configuration module. This allows us to use the @celery_app.task 
# decorator to offload long-running operations.
from celery_app import celery_app


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
    Mail = None
from database import (
    Case,
    NotificationStatus,
    NotificationChannel,
    NotificationLog,
    UserPreference,
    CaseDeadline,
    log_notification,
    has_notification_been_sent,
    get_notification_template_for_user,
)
from core.template_renderer import render_template, validate_template, TemplateValidationError

# Import debug mode helper
def _is_debug_or_testing_mode() -> bool:
    """Return True when explicit debug/testing flags are enabled."""
    return Config.DEBUG or Config.TESTING

logger = structlog.get_logger(__name__)


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

        if not all([self.account_sid, self.auth_token, self.from_number]):
            logger.warning("Twilio credentials not configured. SMS will be mocked.")
            self.client = None
        else:
            self.client = TwilioClient(self.account_sid, self.auth_token)

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
                if _is_debug_or_testing_mode():
                    logger.info(f"[MOCK SMS] To: {to_number}, Message: {message}")
                    return True, f"mock_sms_{datetime.now().timestamp()}", None
                
                error_msg = "Twilio credentials not configured. SMS delivery skipped."
                logger.warning(error_msg)
                return False, None, error_msg

            message_obj = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number,
            )
            logger.info(f"SMS sent successfully. SID: {message_obj.sid}")
            return True, message_obj.sid, None

        except Exception as e:
            error_msg = f"Failed to send SMS: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg


class EmailClient:
    """Wrapper for SendGrid email client"""

    def __init__(self):
        self.api_key = Config.get_sendgrid_api_key()
        self.from_email = Config.SENDGRID_FROM_EMAIL

        if not self.api_key:
            logger.warning("SendGrid API key not configured. Emails will be mocked.")
            self.client = None
        else:
            self.client = SendGridAPIClient(self.api_key)

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
                if _is_debug_or_testing_mode():
                    logger.info(f"[MOCK EMAIL] To: {to_email}, Subject: {subject}")
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
            response = self.client.send(message)
            logger.info(f"Email sent successfully. Status: {response.status_code}")
            return True, response.headers.get("X-Message-ID", "unknown"), None

        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg


# ============================================================================
# ASYNCHRONOUS BACKGROUND TASKS
# ============================================================================

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
    
    This task offloads the synchronous network call to SendGrid to a background 
    worker. It also handles logging the result back to the database to ensure 
    that we maintain an accurate audit trail of all notifications sent.
    
    Args:
        self: The task instance (for retries).
        to_email (str): Recipient email address.
        subject (str): Email subject line.
        html_content (str): The rendered HTML body of the email.
        deadline_id (int, optional): ID of the deadline for logging.
        user_id (int, optional): ID of the user for logging.
        days_left (int, optional): The reminder threshold (e.g., 30, 10, 3, 1).
        
    Returns:
        dict: A summary of the operation results.
    """
    from database import db_session, log_notification, NotificationStatus, NotificationChannel
    
    logger.info(
        "Starting background email delivery", 
        recipient=to_email, 
        subject=subject,
        task_id=self.request.id
    )
    
    # Initialize the EmailClient. We do this inside the task to ensure 
    # that any environment-specific configuration is picked up correctly 
    # by the worker process.
    client = EmailClient()
    
    # Execute the actual network request to SendGrid
    success, message_id, error = client.send_email(to_email, subject, html_content)
    
    # Determine the status for database logging
    status = NotificationStatus.SENT if success else NotificationStatus.FAILED
    
    # If logging metadata was provided, persist the result to the database
    if deadline_id is not None and user_id is not None and days_left is not None:
        try:
            # We use the db_session context manager to ensure the connection 
            # is properly closed and the transaction is committed.
            with db_session() as db:
                log_notification(
                    db=db,
                    deadline_id=deadline_id,
                    user_id=user_id,
                    channel=NotificationChannel.EMAIL,
                    recipient=to_email,
                    days_before=days_left,
                    status=status,
                    message_id=message_id,
                    error_message=error,
                    message_preview=html_content,
                )
                logger.info("Background notification logged successfully", deadline_id=deadline_id)
        except Exception as e:
            logger.error("Failed to log background notification", error=str(e), deadline_id=deadline_id)
    
    # Handle retries if the email failed and we haven't exceeded the limit.
    # We only retry for potentially transient errors.
    if not success and self.request.retries < self.max_retries:
        logger.warning(
            "Email delivery failed, scheduling retry", 
            error=error, 
            retry_count=self.request.retries + 1
        )
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
        self.base_url = Config.BASE_URL.rstrip('/')

    def build_sms_message(self, case_title: str, days_left: int, deadline_date: datetime) -> str:
        """Build SMS reminder message"""
        formatted_date = deadline_date.strftime("%d %b %Y")
        return (
            f"⚖️ LegalAssist: Case '{case_title}' has a deadline in {days_left} day(s). "
            f"Deadline: {formatted_date}. Log in to check details."
        )

    def build_email_message(self, deadline: CaseDeadline, days_left: int) -> Tuple[str, str]:
        """
        Build a premium email reminder content.
        Uses modern HTML/CSS with glassmorphism-inspired design.
        Returns: (subject, html_content)
        """
        formatted_date = deadline.deadline_date.strftime("%d %B %Y")
        escaped_title = html.escape(deadline.case_title)
        escaped_type = html.escape(deadline.deadline_type.title())
        escaped_desc = html.escape(deadline.description) if deadline.description else "No additional details provided."
        
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

        subject = f"⚖️ {urgency_label}: {deadline.case_title} - {escaped_type} Deadline"
        
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

    def send_sms_reminder(
        self,
        db: Session,
        deadline: CaseDeadline,
        user_preference: UserPreference,
        days_left: int,
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

        # Try per-user template first
        message = None
        try:
            tmpl = get_notification_template_for_user(db, deadline.user_id)
            if tmpl and tmpl.sms_template:
                values = {
                    "case_title": deadline.case_title,
                    "case_number": getattr(deadline, "case_id", ""),
                    "deadline_date": deadline.deadline_date.strftime("%d %b %Y") if deadline.deadline_date else "",
                    "days_left": days_left,
                    "court": "",
                    "deadline_type": deadline.deadline_type,
                    "deadline_description": deadline.description or "",
                    "link": f"https://legalassist.ai/cases/{deadline.case_id}",
                }
                message = render_template(tmpl.sms_template, values)
        except TemplateValidationError as e:
            logger.warning("User SMS template invalid, falling back to default: %s", str(e))
        except Exception:
            logger.exception("Error rendering user SMS template; falling back to default")

        if message is None:
            message = self.build_sms_message(deadline.case_title, days_left, deadline.deadline_date)
        success, message_id, error = self.sms_client.send_sms(user_preference.phone_number, message)

        status = NotificationStatus.SENT if success else NotificationStatus.FAILED

        log_notification(
            db=db,
            deadline_id=deadline.id,
            user_id=deadline.user_id,
            channel=NotificationChannel.SMS,
            recipient=user_preference.phone_number,
            days_before=days_left,
            status=status,
            message_id=message_id,
            error_message=error,
            message_preview=message,
        )

        return NotificationResult(
            success=success,
            channel=NotificationChannel.SMS,
            recipient=user_preference.phone_number,
            message_id=message_id,
            error=error,
        )

    def send_email_reminder(
        self,
        db: Session,
        deadline: CaseDeadline,
        user_preference: UserPreference,
        days_left: int,
    ) -> NotificationResult:
        """Send email reminder for a deadline"""
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
                    subject = render_template(tmpl.email_subject_template, values)
                if tmpl.email_html_template:
                    html_content = render_template(tmpl.email_html_template, values)
        except TemplateValidationError as e:
            logger.warning("User email template invalid, falling back to default: %s", str(e))
        except Exception:
            logger.exception("Error rendering user email template; falling back to default")

        if subject is None or html_content is None:
            subject, html_content = self.build_email_message(deadline, days_left)

        # ====================================================================
        # ASYNCHRONOUS DELIVERY OFFLOAD
        # ====================================================================
        # Instead of calling self.email_client.send_email() directly, which 
        # would block the current thread for several seconds while waiting 
        # for the SendGrid API response, we dispatch a Celery task.
        #
        # This allows the request (or the periodic check) to complete 
        # immediately, providing a much smoother and "snappier" experience 
        # for the end-user or the system scheduler.
        # ====================================================================
        
        logger.info(
            "Offloading email delivery to background task", 
            user_id=deadline.user_id,
            deadline_id=deadline.id,
            days_left=days_left
        )
        
        # We use .delay() to send the task to the Redis broker. 
        # The background worker will pick it up and execute it.
        task_result = send_email_task.delay(
            to_email=user_preference.email,
            subject=subject,
            html_content=html_content,
            deadline_id=deadline.id,
            user_id=deadline.user_id,
            days_left=days_left
        )
        
        # We return a successful NotificationResult immediately, noting 
        # that the message ID is the Celery Task ID until the actual 
        # email is processed.
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


        for channel in channels:
            # Check if reminder was already sent for this specific threshold and channel
            if not has_notification_been_sent(db, deadline.id, days_left, channel):
                if channel == NotificationChannel.SMS:
                    result = self.send_sms_reminder(db, deadline, user_preference, days_left)
                    results.append(result)
                elif channel == NotificationChannel.EMAIL:
                    result = self.send_email_reminder(db, deadline, user_preference, days_left)
                    results.append(result)
            else:
                logger.debug("Notification already sent", 
                            channel=channel.value, 
                            days_left=days_left, 
                            deadline_id=deadline.id)

        return results
