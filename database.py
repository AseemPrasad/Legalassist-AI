"""
Database models for deadline tracking and notification management.
Uses SQLAlchemy ORM with SQLite for persistence.
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Enum as SQLEnum,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
import enum
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./legalassist.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class NotificationStatus(str, enum.Enum):
    """Status of sent notifications"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"


class NotificationChannel(str, enum.Enum):
    """Channel for sending notifications"""
    SMS = "sms"
    EMAIL = "email"
    BOTH = "both"


class CaseDeadline(Base):
    """Model for case deadlines"""
    __tablename__ = "case_deadlines"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    case_id = Column(String, nullable=False)
    case_title = Column(String, nullable=False)
    deadline_date = Column(DateTime, nullable=False, index=True)
    deadline_type = Column(String, nullable=False)  # appeal, filing, submission, etc.
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    is_completed = Column(Boolean, default=False)

    # Relationships
    notifications = relationship("NotificationLog", back_populates="deadline", cascade="all, delete-orphan")

    def days_until_deadline(self) -> int:
        """Calculate days remaining until deadline"""
        now = datetime.now(timezone.utc)
        delta = self.deadline_date - now
        return max(0, delta.days)

    def __repr__(self):
        return f"<CaseDeadline(user_id={self.user_id}, case_id={self.case_id}, deadline_date={self.deadline_date})>"


class UserPreference(Base):
    """Model for user notification preferences"""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, nullable=False, index=True)
    phone_number = Column(String, nullable=True)
    email = Column(String, nullable=False)
    notification_channel = Column(SQLEnum(NotificationChannel), default=NotificationChannel.BOTH)
    timezone = Column(String, default="UTC")  # e.g., "Asia/Kolkata", "America/New_York"
    notify_30_days = Column(Boolean, default=True)
    notify_10_days = Column(Boolean, default=True)
    notify_3_days = Column(Boolean, default=True)
    notify_1_day = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id}, channel={self.notification_channel})>"


class NotificationLog(Base):
    """Model for tracking sent notifications"""
    __tablename__ = "notification_logs"

    id = Column(Integer, primary_key=True, index=True)
    deadline_id = Column(Integer, ForeignKey("case_deadlines.id"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    channel = Column(SQLEnum(NotificationChannel), nullable=False)
    status = Column(SQLEnum(NotificationStatus), default=NotificationStatus.PENDING, index=True)
    recipient = Column(String, nullable=False)  # phone or email
    days_before = Column(Integer, nullable=False)  # 30, 10, 3, or 1 day reminder
    message_id = Column(String, nullable=True)  # From Twilio or SendGrid
    error_message = Column(Text, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    deadline = relationship("CaseDeadline", back_populates="notifications")

    def __repr__(self):
        return f"<NotificationLog(user_id={self.user_id}, status={self.status}, channel={self.channel})>"


class CaseRecord(Base):
    """Model for tracking individual case records (anonymized)"""
    __tablename__ = "case_records"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, unique=True, nullable=False, index=True)  # Hashed ID for privacy
    case_type = Column(String, nullable=False, index=True)  # civil, criminal, family, etc.
    jurisdiction = Column(String, nullable=False, index=True)  # Delhi, Maharashtra, etc.
    court_name = Column(String, nullable=True, index=True)  # District court, High court, etc.
    judge_name = Column(String, nullable=True, index=True)  # Anonymized judge reference
    plaintiff_type = Column(String, nullable=True)  # individual, organization, government
    defendant_type = Column(String, nullable=True)
    case_value = Column(String, nullable=True)  # value range: <1L, 1-5L, 5-10L, >10L
    outcome = Column(String, nullable=False, index=True)  # plaintiff_won, defendant_won, settlement, dismissal
    judgment_summary = Column(Text, nullable=True)  # Brief summary of judgment
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    outcome_data = relationship("CaseOutcome", back_populates="case_record", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CaseRecord(case_type={self.case_type}, jurisdiction={self.jurisdiction}, outcome={self.outcome})>"


class CaseOutcome(Base):
    """Model for tracking appeal outcomes and follow-ups"""
    __tablename__ = "case_outcomes"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("case_records.id"), nullable=False, unique=True, index=True)
    appeal_filed = Column(Boolean, default=False, nullable=False)
    appeal_date = Column(DateTime, nullable=True)
    appeal_outcome = Column(String, nullable=True)  # appeal_allowed, appeal_rejected, withdrawn, pending
    appeal_success = Column(Boolean, nullable=True)  # True = won, False = lost, None = pending
    time_to_appeal_verdict = Column(Integer, nullable=True)  # days
    appeal_cost = Column(String, nullable=True)  # estimated cost range
    additional_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    case_record = relationship("CaseRecord", back_populates="outcome_data")

    def __repr__(self):
        return f"<CaseOutcome(case_id={self.case_id}, appeal_filed={self.appeal_filed}, appeal_success={self.appeal_success})>"


class CaseAnalytics(Base):
    """Model for aggregated analytics (refreshed periodically)"""
    __tablename__ = "case_analytics"

    id = Column(Integer, primary_key=True, index=True)
    case_type = Column(String, nullable=False)  # civil, criminal, etc.
    jurisdiction = Column(String, nullable=False, index=True)
    court_name = Column(String, nullable=True)
    judge_name = Column(String, nullable=True)
    
    # Metrics
    total_cases = Column(Integer, default=0)
    plaintiff_win_count = Column(Integer, default=0)
    defendant_win_count = Column(Integer, default=0)
    settlement_count = Column(Integer, default=0)
    
    appeals_filed = Column(Integer, default=0)
    appeals_successful = Column(Integer, default=0)
    appeal_success_rate = Column(String, default="0%")  # e.g., "22%"
    
    avg_case_duration = Column(Integer, nullable=True)  # days
    avg_appeal_duration = Column(Integer, nullable=True)  # days
    avg_appeal_cost = Column(Integer, nullable=True)  # rupees
    
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<CaseAnalytics(jurisdiction={self.jurisdiction}, appeal_success_rate={self.appeal_success_rate})>"


class UserFeedback(Base):
    """Model for tracking user feedback on case outcomes"""
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("case_records.id"), nullable=True)
    
    # Feedback fields
    did_appeal = Column(Boolean, nullable=True)
    appeal_outcome = Column(String, nullable=True)  # won, lost, pending, withdrawn
    appeal_cost = Column(Integer, nullable=True)  # actual cost in rupees
    time_to_verdict = Column(Integer, nullable=True)  # days
    case_type = Column(String, nullable=True)
    jurisdiction = Column(String, nullable=True)
    
    # Satisfaction feedback
    satisfaction_rating = Column(Integer, nullable=True)  # 1-5
    feedback_text = Column(Text, nullable=True)  # User's notes
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<UserFeedback(user_id={self.user_id}, appeal_outcome={self.appeal_outcome})>"


# Database initialization
def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


# ==================== Helper Functions ====================


def create_or_update_user_preference(
    db: Session,
    user_id: str,
    email: str,
    phone_number: Optional[str] = None,
    notification_channel: NotificationChannel = NotificationChannel.BOTH,
    timezone: str = "UTC",
) -> UserPreference:
    """Create or update user notification preferences"""
    pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    
    if pref:
        pref.email = email
        pref.phone_number = phone_number
        pref.notification_channel = notification_channel
        pref.timezone = timezone
        pref.updated_at = datetime.now(timezone=timezone.utc)
    else:
        pref = UserPreference(
            user_id=user_id,
            email=email,
            phone_number=phone_number,
            notification_channel=notification_channel,
            timezone=timezone,
        )
        db.add(pref)
    
    db.commit()
    db.refresh(pref)
    return pref


def create_case_deadline(
    db: Session,
    user_id: str,
    case_id: str,
    case_title: str,
    deadline_date: datetime,
    deadline_type: str,
    description: Optional[str] = None,
) -> CaseDeadline:
    """Create a new case deadline"""
    deadline = CaseDeadline(
        user_id=user_id,
        case_id=case_id,
        case_title=case_title,
        deadline_date=deadline_date,
        deadline_type=deadline_type,
        description=description,
    )
    db.add(deadline)
    db.commit()
    db.refresh(deadline)
    return deadline


def get_upcoming_deadlines(db: Session, days_before: int = 30) -> List[CaseDeadline]:
    """Get all deadlines that are X days away"""
    now = datetime.now(timezone.utc)
    target_date = datetime.fromtimestamp(now.timestamp() + (days_before * 86400), tz=timezone.utc)
    
    return db.query(CaseDeadline).filter(
        CaseDeadline.is_completed == False,
        CaseDeadline.deadline_date <= target_date,
        CaseDeadline.deadline_date > now,
    ).all()


def get_user_deadlines(db: Session, user_id: str) -> List[CaseDeadline]:
    """Get all active deadlines for a user"""
    now = datetime.now(timezone.utc)
    return db.query(CaseDeadline).filter(
        CaseDeadline.user_id == user_id,
        CaseDeadline.is_completed == False,
        CaseDeadline.deadline_date > now,
    ).order_by(CaseDeadline.deadline_date).all()


def has_notification_been_sent(
    db: Session,
    deadline_id: int,
    days_before: int,
    channel: NotificationChannel,
) -> bool:
    """Check if a notification was already sent for this deadline"""
    return db.query(NotificationLog).filter(
        NotificationLog.deadline_id == deadline_id,
        NotificationLog.days_before == days_before,
        NotificationLog.channel == channel,
        NotificationLog.status.in_([NotificationStatus.SENT, NotificationStatus.OPENED]),
    ).first() is not None


def log_notification(
    db: Session,
    deadline_id: int,
    user_id: str,
    channel: NotificationChannel,
    recipient: str,
    days_before: int,
    status: NotificationStatus = NotificationStatus.PENDING,
    message_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> NotificationLog:
    """Log a notification attempt"""
    log = NotificationLog(
        deadline_id=deadline_id,
        user_id=user_id,
        channel=channel,
        recipient=recipient,
        days_before=days_before,
        status=status,
        message_id=message_id,
        error_message=error_message,
        sent_at=datetime.now(timezone.utc) if status != NotificationStatus.PENDING else None,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_notification_history(db: Session, user_id: str, limit: int = 50) -> List[NotificationLog]:
    """Get notification history for a user"""
    return db.query(NotificationLog).filter(
        NotificationLog.user_id == user_id
    ).order_by(NotificationLog.created_at.desc()).limit(limit).all()


# ==================== Analytics & Case Tracking Helper Functions ====================


def create_case_record(
    db: Session,
    case_id: str,
    case_type: str,
    jurisdiction: str,
    court_name: Optional[str] = None,
    judge_name: Optional[str] = None,
    plaintiff_type: Optional[str] = None,
    defendant_type: Optional[str] = None,
    case_value: Optional[str] = None,
    outcome: str = "pending",
    judgment_summary: Optional[str] = None,
) -> CaseRecord:
    """Create a new case record for analytics"""
    case = CaseRecord(
        case_id=case_id,
        case_type=case_type,
        jurisdiction=jurisdiction,
        court_name=court_name,
        judge_name=judge_name,
        plaintiff_type=plaintiff_type,
        defendant_type=defendant_type,
        case_value=case_value,
        outcome=outcome,
        judgment_summary=judgment_summary,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def update_case_outcome(
    db: Session,
    case_id: str,
    appeal_filed: bool = False,
    appeal_date: Optional[datetime] = None,
    appeal_outcome: Optional[str] = None,
    appeal_success: Optional[bool] = None,
    time_to_appeal_verdict: Optional[int] = None,
    appeal_cost: Optional[str] = None,
) -> CaseOutcome:
    """Update case outcome with appeal information"""
    case = db.query(CaseRecord).filter(CaseRecord.case_id == case_id).first()
    if not case:
        raise ValueError(f"Case {case_id} not found")
    
    outcome = db.query(CaseOutcome).filter(CaseOutcome.case_id == case.id).first()
    if not outcome:
        outcome = CaseOutcome(case_id=case.id)
        db.add(outcome)
    
    outcome.appeal_filed = appeal_filed
    if appeal_date:
        outcome.appeal_date = appeal_date
    if appeal_outcome:
        outcome.appeal_outcome = appeal_outcome
    if appeal_success is not None:
        outcome.appeal_success = appeal_success
    if time_to_appeal_verdict:
        outcome.time_to_appeal_verdict = time_to_appeal_verdict
    if appeal_cost:
        outcome.appeal_cost = appeal_cost
    
    db.commit()
    db.refresh(outcome)
    return outcome


def get_case_record(db: Session, case_id: str) -> Optional[CaseRecord]:
    """Get a case record by ID"""
    return db.query(CaseRecord).filter(CaseRecord.case_id == case_id).first()


def get_cases_by_criteria(
    db: Session,
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    court_name: Optional[str] = None,
    judge_name: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = 100,
) -> List[CaseRecord]:
    """Get cases matching specific criteria"""
    query = db.query(CaseRecord)
    
    if case_type:
        query = query.filter(CaseRecord.case_type == case_type)
    if jurisdiction:
        query = query.filter(CaseRecord.jurisdiction == jurisdiction)
    if court_name:
        query = query.filter(CaseRecord.court_name == court_name)
    if judge_name:
        query = query.filter(CaseRecord.judge_name == judge_name)
    if outcome:
        query = query.filter(CaseRecord.outcome == outcome)
    
    return query.order_by(CaseRecord.created_at.desc()).limit(limit).all()


def submit_user_feedback(
    db: Session,
    user_id: str,
    did_appeal: Optional[bool] = None,
    appeal_outcome: Optional[str] = None,
    appeal_cost: Optional[int] = None,
    time_to_verdict: Optional[int] = None,
    case_type: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    satisfaction_rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
) -> UserFeedback:
    """Submit feedback from user about case outcome"""
    feedback = UserFeedback(
        user_id=user_id,
        did_appeal=did_appeal,
        appeal_outcome=appeal_outcome,
        appeal_cost=appeal_cost,
        time_to_verdict=time_to_verdict,
        case_type=case_type,
        jurisdiction=jurisdiction,
        satisfaction_rating=satisfaction_rating,
        feedback_text=feedback_text,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_user_feedback(db: Session, user_id: str, limit: int = 50) -> List[UserFeedback]:
    """Get feedback submitted by a user"""
    return db.query(UserFeedback).filter(
        UserFeedback.user_id == user_id
    ).order_by(UserFeedback.created_at.desc()).limit(limit).all()

