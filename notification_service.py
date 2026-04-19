"""
Notification service for sending SMS and Email reminders using Twilio and SendGrid.
Handles delivery tracking and retry logic.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

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
    NotificationStatus,
    NotificationChannel,
    NotificationLog,
    UserPreference,
    CaseDeadline,
    log_notification,
)

logger = logging.getLogger(__name__)


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

    def build_email_message(self, case_title: str, days_left: int, deadline_date: datetime, case_id: str) -> Tuple[str, str]:
        """
        Build email reminder content.
        Returns: (subject, html_content)
        """
        formatted_date = deadline_date.strftime("%d %B %Y")
        
        subject = f"⚖️ Urgent: {case_title} - Deadline in {days_left} day(s)"
        
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #1a5490;">⚖️ Deadline Reminder</h2>
                    
                    <p style="font-size: 16px;">Dear Litigant,</p>
                    
                    <p style="font-size: 16px;">
                        Your case <strong>"{case_title}"</strong> has an important deadline approaching.
                    </p>
                    
                    <div style="background-color: #fff3cd; border-left: 4px solid #ff9800; padding: 15px; margin: 20px 0;">
                        <p style="margin: 0; font-size: 18px; font-weight: bold;">
                            ⏰ Deadline: <span style="color: #d32f2f;">{formatted_date}</span>
                        </p>
                        <p style="margin: 5px 0; font-size: 16px;">
                            <strong>{days_left} day(s) remaining</strong>
                        </p>
                    </div>
                    
                    <p style="font-size: 16px;">
                        Missing this deadline could result in case closure or dismissal. 
                        Please take action immediately.
                    </p>
                    
                    <div style="margin: 30px 0;">
                        <a href="https://legalassist.ai/cases/{case_id}" 
                           style="background-color: #1a5490; color: white; padding: 12px 30px; 
                                  text-decoration: none; border-radius: 5px; display: inline-block;
                                  font-weight: bold;">
                            View Case Details
                        </a>
                    </div>
                    
                    <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                    
                    <p style="font-size: 12px; color: #666;">
                        This is an automated reminder from LegalAssist AI. 
                        You can manage your notification preferences in your account settings.
                    </p>
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
        
        subject, html_content = self.build_email_message(
            deadline.case_title, days_left, deadline.deadline_date, deadline.case_id
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
    ) -> List[NotificationResult]:
        """
        Send appropriate reminders based on days until deadline and user preferences.
        Checks which reminders should be sent for 30, 10, 3, and 1 day marks.
        """
        results = []
        days_left = deadline.days_until_deadline()

        # Determine which reminders should be sent
        reminder_thresholds = []
        if days_left == 30 and user_preference.notify_30_days:
            reminder_thresholds.append(30)
        elif days_left == 10 and user_preference.notify_10_days:
            reminder_thresholds.append(10)
        elif days_left == 3 and user_preference.notify_3_days:
            reminder_thresholds.append(3)
        elif days_left == 1 and user_preference.notify_1_day:
            reminder_thresholds.append(1)

        for threshold in reminder_thresholds:
            # Send based on user's notification channel preference
            if user_preference.notification_channel in [NotificationChannel.SMS, NotificationChannel.BOTH]:
                from database import has_notification_been_sent
                
                if not has_notification_been_sent(db, deadline.id, threshold, NotificationChannel.SMS):
                    result = self.send_sms_reminder(db, deadline, user_preference, threshold)
                    results.append(result)

            if user_preference.notification_channel in [NotificationChannel.EMAIL, NotificationChannel.BOTH]:
                from database import has_notification_been_sent
                
                if not has_notification_been_sent(db, deadline.id, threshold, NotificationChannel.EMAIL):
                    result = self.send_email_reminder(db, deadline, user_preference, threshold)
                    results.append(result)

        return results
