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
    template_variants = Column(JSON, nullable=True)
    sms_template = Column(Text, nullable=True)
    email_subject_template = Column(String(255), nullable=True)
    email_html_template = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    def _template_scope_key(self, value):
        if value is None:
            return "default"
        if hasattr(value, "value"):
            value = value.value
        text = str(value).strip().lower()
        return text or "default"

    def resolve_templates(self, channel=None, language=None):
        variants = self.template_variants if isinstance(self.template_variants, dict) else {}
        channel_key = self._template_scope_key(channel)
        language_key = self._template_scope_key(language)

        candidate_maps = []
        for candidate_channel, candidate_language in (
            (channel_key, language_key),
            (channel_key, "default"),
            ("default", language_key),
            ("default", "default"),
        ):
            channel_bucket = variants.get(candidate_channel)
            if isinstance(channel_bucket, dict):
                candidate = channel_bucket.get(candidate_language)
                if isinstance(candidate, dict):
                    candidate_maps.append(candidate)

        selected = candidate_maps[0] if candidate_maps else {}
        return {
            "sms_template": selected.get("sms_template") or self.sms_template,
            "email_subject_template": selected.get("email_subject_template") or self.email_subject_template,
            "email_html_template": selected.get("email_html_template") or self.email_html_template,
        }

    def set_template_variant(
        self,
        channel=None,
        language=None,
        sms_template=None,
        email_subject_template=None,
        email_html_template=None,
    ):
        variants = self.template_variants if isinstance(self.template_variants, dict) else {}
        variants = dict(variants)

        channel_key = self._template_scope_key(channel)
        language_key = self._template_scope_key(language)

        channel_bucket = dict(variants.get(channel_key, {}))
        variant = dict(channel_bucket.get(language_key, {}))

        if sms_template is not None:
            variant["sms_template"] = sms_template
        if email_subject_template is not None:
            variant["email_subject_template"] = email_subject_template
        if email_html_template is not None:
            variant["email_html_template"] = email_html_template

        channel_bucket[language_key] = variant
        variants[channel_key] = channel_bucket
        self.template_variants = variants


class NotificationLog(Base):
    __tablename__ = "notification_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "deadline_id", "days_before", "channel", name="uq_notification_user_deadline_days_channel"),
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
