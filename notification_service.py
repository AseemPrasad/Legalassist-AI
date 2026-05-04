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
)

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
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")

        if not all([self.account_sid, self.auth_token, self.from_number]):
            logger.warning("Twilio credentials not configured. SMS will be mocked.")
            self.client = None
        else:
            self.client = TwilioClient(self.account_sid, self.auth_token)

    def send_sms(self, to_number: str, message: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Send SMS message.
        Returns: (success, message_id, error)
        """
        try:
            if not self.client:
                # Mock mode - log instead of sending
                logger.info(f"[MOCK SMS] To: {to_number}, Message: {message}")
                return True, f"mock_sms_{datetime.now().timestamp()}", None

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
        self.api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("SENDGRID_FROM_EMAIL", "noreply@legalassist.ai")

        if not self.api_key:
            logger.warning("SendGrid API key not configured. Emails will be mocked.")
            self.client = None
        else:
            self.client = SendGridAPIClient(self.api_key)

    def send_email(self, to_email: str, subject: str, html_content: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Send email.
        Returns: (success, message_id, error)
        """
        try:
            if not self.client:
                # Mock mode - log instead of sending
                logger.info(f"[MOCK EMAIL] To: {to_email}, Subject: {subject}")
                return True, f"mock_email_{datetime.now().timestamp()}", None

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


class NotificationService:
    """Main service for sending deadline reminders"""

    def __init__(self):
        self.sms_client = SMSClient()
        self.email_client = EmailClient()

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
                        <a href="https://legalassist.ai/cases/{deadline.case_id}" class="cta-button">
                            View Case Dashboard
                        </a>
                    </div>
                </div>
                <div class="footer">
                    <p>This is an automated notification from your LegalAssist AI account.<br>
                    Missing deadlines can lead to dismissal of your case. Please consult with your legal counsel immediately.</p>
                    <p>Manage your <a href="https://legalassist.ai/settings">Notification Preferences</a></p>
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
        case = db.query(Case).filter(Case.id == deadline.case_id).first()
        case_number = case.case_number if case else str(deadline.case_id)

        subject, html_content = self.build_email_message(
            deadline.case_title, days_left, deadline.deadline_date, case_number
        )
        success, message_id, error = self.email_client.send_email(
            user_preference.email, subject, html_content
        )

        status = NotificationStatus.SENT if success else NotificationStatus.FAILED

        log_notification(
            db=db,
            deadline_id=deadline.id,
            user_id=deadline.user_id,
            channel=NotificationChannel.EMAIL,
            recipient=user_preference.email,
            days_before=days_left,
            status=status,
            message_id=message_id,
            error_message=error,
        )

        return NotificationResult(
            success=success,
            channel=NotificationChannel.EMAIL,
            recipient=user_preference.email,
            message_id=message_id,
            error=error,
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

        from database import has_notification_been_sent

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
