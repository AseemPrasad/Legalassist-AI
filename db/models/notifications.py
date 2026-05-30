import datetime as dt
import enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Enum as SQLEnum, Index, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from db.base import Base


class NotificationStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"


class NotificationChannel(str, enum.Enum):
    SMS = "sms"
    EMAIL = "email"
    BOTH = "both"


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    phone_number = Column(String(255), nullable=True)
    email = Column(String(255), nullable=False)
    notification_channel = Column(SQLEnum(NotificationChannel), default=NotificationChannel.BOTH)
    timezone = Column(String(255), default="UTC")
    notify_30_days = Column(Boolean, default=True)
    notify_10_days = Column(Boolean, default=True)
    notify_3_days = Column(Boolean, default=True)
    notify_1_day = Column(Boolean, default=True)
    holiday_aware_reminders = Column(Boolean, default=False)
    holiday_country = Column(String(255), nullable=True)
    holiday_region = Column(String(255), nullable=True)
    holiday_calendar_json = Column(Text, nullable=True)
    reminder_thresholds = Column(JSON, default=lambda: [30, 10, 3, 1], nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    user = relationship("db.models.auth.User", back_populates="preferences")

    def get_reminder_thresholds(self) -> list[int]:
        if self.reminder_thresholds is not None:
            if isinstance(self.reminder_thresholds, list):
                return [int(x) for x in self.reminder_thresholds]
            elif isinstance(self.reminder_thresholds, str):
                try:
                    import json
                    parsed = json.loads(self.reminder_thresholds)
                    if isinstance(parsed, list):
                        return [int(x) for x in parsed]
                except Exception:
                    pass
                try:
                    return [int(x.strip()) for x in self.reminder_thresholds.split(",") if x.strip().isdigit()]
                except Exception:
                    pass
        thresholds = []
        if getattr(self, "notify_30_days", True):
            thresholds.append(30)
        if getattr(self, "notify_10_days", True):
            thresholds.append(10)
        if getattr(self, "notify_3_days", True):
            thresholds.append(3)
        if getattr(self, "notify_1_day", True):
            thresholds.append(1)
        return thresholds


class NotificationTemplate(Base):
    __tablename__ = "notification_templates"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    sms_template = Column(Text, nullable=True)
    email_subject_template = Column(String(255), nullable=True)
    email_html_template = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))


class NotificationLog(Base):
    __tablename__ = "notification_logs"
    __table_args__ = (
        UniqueConstraint("deadline_id", "days_before", "channel", name="uq_notification_deadline_days_channel"),
    )

    id = Column(Integer, primary_key=True)
    deadline_id = Column(Integer, ForeignKey("case_deadlines.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    channel = Column(SQLEnum(NotificationChannel), nullable=False)
    status = Column(SQLEnum(NotificationStatus), default=NotificationStatus.PENDING, index=True)
    attempted_channels = Column(JSON, nullable=True)
    recipient = Column(String(255), nullable=False)
    days_before = Column(Integer, nullable=False)
    message_id = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    message_preview = Column(Text, nullable=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    deadline = relationship("db.models.cases.CaseDeadline", back_populates="notifications")
