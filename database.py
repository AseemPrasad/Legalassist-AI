"""
Database models for deadline tracking and notification management.
Uses SQLAlchemy ORM with SQLite for persistence.
"""

import datetime as dt
import logging
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
    JSON,
    UniqueConstraint,
    Index,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
import enum
from contextlib import contextmanager
from config import Config

# Database setup
DATABASE_URL = Config.DATABASE_URL
_db_url = make_url(DATABASE_URL)
_is_sqlite = _db_url.get_backend_name() == "sqlite"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if _is_sqlite else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)
Base = declarative_base()

logger = logging.getLogger(__name__)


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

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    case_title = Column(String(255), nullable=False)
    deadline_date = Column(DateTime(timezone=True), nullable=False, index=True)
    deadline_type = Column(String(255), nullable=False)  # appeal, filing, submission, etc.
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))
    is_completed = Column(Boolean, default=False, index=True)

    # Relationships
    case = relationship("Case", back_populates="deadlines")
    notifications = relationship("NotificationLog", back_populates="deadline", cascade="all, delete-orphan")
    attachments = relationship("Attachment", back_populates="deadline", cascade="all, delete-orphan")

    def days_until_deadline(self) -> int:
        """Calculate days remaining until deadline"""
        now = dt.datetime.now(dt.timezone.utc)
        deadline = self.deadline_date
        if deadline and deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=dt.timezone.utc)
        delta = deadline - now
        return max(0, delta.days)

    def __repr__(self):
        return f"<CaseDeadline(user_id={self.user_id}, case_id={self.case_id}, deadline_date={self.deadline_date})>"


class UserPreference(Base):
    """Model for user notification preferences"""
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    phone_number = Column(String(255), nullable=True)
    email = Column(String(255), nullable=False)
    notification_channel = Column(SQLEnum(NotificationChannel), default=NotificationChannel.BOTH)
    timezone = Column(String(255), default="UTC")  # e.g., "Asia/Kolkata", "America/New_York"
    notify_30_days = Column(Boolean, default=True)
    notify_10_days = Column(Boolean, default=True)
    notify_3_days = Column(Boolean, default=True)
    notify_1_day = Column(Boolean, default=True)

    # Holiday-aware reminder engine (MVP)
    holiday_aware_reminders = Column(Boolean, default=False)
    holiday_country = Column(String(255), nullable=True)  # e.g., "IN" (optional in MVP)
    holiday_region = Column(String(255), nullable=True)   # e.g., "MH" / state/province (optional in MVP)
    # JSON array of ISO dates: ["2026-01-26", "2026-03-29", ...]
    holiday_calendar_json = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))


    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id}, channel={self.notification_channel})>"


class NotificationLog(Base):
    """Model for tracking sent notifications"""
    __tablename__ = "notification_logs"

    id = Column(Integer, primary_key=True)
    deadline_id = Column(Integer, ForeignKey("case_deadlines.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    channel = Column(SQLEnum(NotificationChannel), nullable=False)
    status = Column(SQLEnum(NotificationStatus), default=NotificationStatus.PENDING, index=True)
    recipient = Column(String(255), nullable=False)  # phone or email
    days_before = Column(Integer, nullable=False)  # 30, 10, 3, or 1 day reminder
    message_id = Column(String(255), nullable=True)  # From Twilio or SendGrid
    error_message = Column(Text, nullable=True)
    message_preview = Column(Text, nullable=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    # Relationships
    deadline = relationship("CaseDeadline", back_populates="notifications")

    def __repr__(self):
        return f"<NotificationLog(user_id={self.user_id}, status={self.status}, channel={self.channel})>"


class CaseRecord(Base):
    """Model for tracking individual case records (anonymized)"""
    __tablename__ = "case_records"

    id = Column(Integer, primary_key=True)
    hashed_case_id = Column(String(255), unique=True, nullable=False, index=True)  # Hashed ID for privacy
    case_type = Column(String(255), nullable=False, index=True)  # civil, criminal, family, etc.
    jurisdiction = Column(String(255), nullable=False, index=True)  # Delhi, Maharashtra, etc.
    court_name = Column(String(255), nullable=True, index=True)  # District court, High court, etc.
    judge_name = Column(String(255), nullable=True, index=True)  # Anonymized judge reference
    plaintiff_type = Column(String(255), nullable=True)  # individual, organization, government
    defendant_type = Column(String(255), nullable=True)
    case_value = Column(String(255), nullable=True)  # value range: <1L, 1-5L, 5-10L, >10L
    outcome = Column(String(255), nullable=False, index=True)  # plaintiff_won, defendant_won, settlement, dismissal
    judgment_summary = Column(Text, nullable=True)  # Brief summary of judgment
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationships
    outcome_data = relationship("CaseOutcome", back_populates="case_record", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CaseRecord(case_type={self.case_type}, jurisdiction={self.jurisdiction}, outcome={self.outcome})>"


class CaseOutcome(Base):
    """Model for tracking appeal outcomes and follow-ups"""
    __tablename__ = "case_outcomes"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("case_records.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    appeal_filed = Column(Boolean, default=False, nullable=False)
    appeal_date = Column(DateTime(timezone=True), nullable=True)
    appeal_outcome = Column(String(255), nullable=True)  # appeal_allowed, appeal_rejected, withdrawn, pending
    appeal_success = Column(Boolean, nullable=True)  # True = won, False = lost, None = pending
    time_to_appeal_verdict = Column(Integer, nullable=True)  # days
    appeal_cost = Column(String(255), nullable=True)  # estimated cost range
    additional_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationships
    case_record = relationship("CaseRecord", back_populates="outcome_data")

    def __repr__(self):
        return f"<CaseOutcome(case_id={self.case_id}, appeal_filed={self.appeal_filed}, appeal_success={self.appeal_success})>"


class CaseAnalytics(Base):
    """Model for aggregated analytics (refreshed periodically)"""
    __tablename__ = "case_analytics"

    id = Column(Integer, primary_key=True)
    case_type = Column(String(255), nullable=False)  # civil, criminal, etc.
    jurisdiction = Column(String(255), nullable=False, index=True)
    court_name = Column(String(255), nullable=True)
    judge_name = Column(String(255), nullable=True)
    
    # Metrics
    total_cases = Column(Integer, default=0)
    plaintiff_win_count = Column(Integer, default=0)
    defendant_win_count = Column(Integer, default=0)
    settlement_count = Column(Integer, default=0)
    
    appeals_filed = Column(Integer, default=0)
    appeals_successful = Column(Integer, default=0)
    appeal_success_rate = Column(String(255), default="0%")  # e.g., "22%"
    
    avg_case_duration = Column(Integer, nullable=True)  # days
    avg_appeal_duration = Column(Integer, nullable=True)  # days
    avg_appeal_cost = Column(Integer, nullable=True)  # rupees
    
    last_updated = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    def __repr__(self):
        return f"<CaseAnalytics(jurisdiction={self.jurisdiction}, appeal_success_rate={self.appeal_success_rate})>"


class NotificationTemplate(Base):
    """Per-user notification templates for SMS and Email"""
    __tablename__ = "notification_templates"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    sms_template = Column(Text, nullable=True)
    email_subject_template = Column(String(255), nullable=True)
    email_html_template = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationship back to user is optional
    def __repr__(self):
        return f"<NotificationTemplate(user_id={self.user_id})>"


class UserFeedback(Base):
    """Model for tracking user feedback on case outcomes"""
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("case_records.id", ondelete="CASCADE"), nullable=True)

    # Feedback fields
    did_appeal = Column(Boolean, nullable=True)
    appeal_outcome = Column(String(255), nullable=True)  # won, lost, pending, withdrawn
    appeal_cost = Column(Integer, nullable=True)  # actual cost in rupees
    time_to_verdict = Column(Integer, nullable=True)  # days
    case_type = Column(String(255), nullable=True)
    jurisdiction = Column(String(255), nullable=True)

    # Satisfaction feedback
    satisfaction_rating = Column(Integer, nullable=True)  # 1-5
    feedback_text = Column(Text, nullable=True)  # User's notes
    
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    def __repr__(self):
        return f"<UserFeedback(user_id={self.user_id}, appeal_outcome={self.appeal_outcome})>"


class ModelFeedback(Base):
    """User feedback on model outputs for later training and evaluation"""
    __tablename__ = "model_feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    task = Column(String(100), nullable=False, index=True)  # summary, remedy, appeal_estimate, etc.
    case_id = Column(Integer, ForeignKey("case_records.id", ondelete="SET NULL"), nullable=True, index=True)
    is_accurate = Column(Boolean, nullable=True, index=True)  # True=accurate, False=inaccurate, None=neutral
    corrected_text = Column(Text, nullable=True)
    feedback_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    case = relationship("CaseRecord")

    def __repr__(self):
        return f"<ModelFeedback(model={self.model_name}, task={self.task}, accurate={self.is_accurate})>"


class ModelPerformance(Base):
    """Aggregated model performance metrics (materialized/updated periodically)"""
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False, index=True)
    task = Column(String(100), nullable=False, index=True)
    case_type = Column(String(100), nullable=True, index=True)
    jurisdiction = Column(String(100), nullable=True, index=True)
    samples = Column(Integer, default=0)
    accurate_count = Column(Integer, default=0)
    accuracy = Column(String(50), default="0%")
    average_latency_ms = Column(Integer, nullable=True)
    average_cost = Column(Integer, nullable=True)  # in cents or smallest currency unit
    last_updated = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    def __repr__(self):
        return f"<ModelPerformance(model={self.model_name}, task={self.task}, accuracy={self.accuracy})>"


class ModelRoutingRule(Base):
    """Rule for routing tasks to specific models based on case attributes"""
    __tablename__ = "model_routing_rule"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    case_type = Column(String(100), nullable=True)
    jurisdiction = Column(String(100), nullable=True)
    min_case_value = Column(String(50), nullable=True)
    task = Column(String(100), nullable=False)
    preferred_model = Column(String(255), nullable=False)
    approved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    def __repr__(self):
        return f"<ModelRoutingRule(name={self.name}, task={self.task}, model={self.preferred_model})>"


class SimilarityFeedback(Base):
    """Model for tracking similarity search relevance feedback"""
    __tablename__ = "similarity_feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    query_signature = Column(String(512), nullable=False, index=True)
    candidate_case_id = Column(Integer, ForeignKey("case_records.id", ondelete="CASCADE"), nullable=False, index=True)
    relevance = Column(Boolean, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    candidate_case = relationship("CaseRecord")

    def __repr__(self):
        return (
            f"<SimilarityFeedback(user_id={self.user_id}, candidate_case_id={self.candidate_case_id}, "
            f"relevance={self.relevance})>"
        )


# ==================== New Models for Case History & Authentication ====================


class User(Base):
    """
    ===========================================================================
    User Model Definition
    ===========================================================================
    
    This model handles all user authentication, authorization, and core profile
    data within the system. It serves as the primary entity to which almost all
    other domain objects are linked (e.g., Cases, Deadlines, Preferences).
    
    Why Indexing Matters Here:
    ---------------------------------------------------------------------------
    User lookups during authentication scale poorly with large datasets. An O(N)
    sequential scan across the users table during a login or token verification
    event creates a significant performance bottleneck and potential vector for
    Denial of Service (DoS) attacks.
    
    By explicitly defining a dedicated database index on the `email` column 
    using the SQLAlchemy `Index` construct, we ensure that database engines 
    like PostgreSQL or MySQL utilize a B-Tree (or similar) index structure. 
    This reduces query execution time for authentication-related lookups from 
    O(N) down to O(log N). 
    
    This is critical because the `email` field is used as the primary lookup
    key during the JWT validation process, the OTP generation process, and 
    the user login flow.
    
    Attributes:
    ---------------------------------------------------------------------------
    id : int
        The primary key for the user. Used as a foreign key in most tables.
    
    email : str
        The user's email address. Must be unique across the system. This 
        field is explicitly indexed to optimize authentication queries.
        
    created_at : datetime
        Timestamp recording when the user was initially created in the system.
        Stored in UTC.
        
    last_login : datetime
        Timestamp of the user's most recent successful authentication event.
        Useful for auditing, session invalidation, and analytics.
        
    is_verified : bool
        Flag indicating whether the user has successfully completed the email
        OTP verification process at least once. Unverified users may have
        restricted access to system features.
        
    Relationships:
    ---------------------------------------------------------------------------
    cases : list[Case]
        A collection of all legal cases associated with this user.
        Cascade behavior ensures that when a user is deleted, all their cases
        are orphaned and subsequently deleted.
        
    preferences : list[UserPreference]
        The user's notification and system preferences.
        Cascade behavior matches the cases relationship.
        
    ===========================================================================
    """

    __tablename__ = "users"

    # -------------------------------------------------------------------------
    # Table Arguments & Indexes
    # -------------------------------------------------------------------------
    # We use a dedicated Index construct here rather than just `index=True` on 
    # the Column definition. This provides several benefits:
    #
    # 1. It explicitly names the index (ix_users_email), making migrations and 
    #    database maintenance operations more predictable and traceable.
    #
    # 2. It allows for future expansion (e.g., adding partial indexes or 
    #    composite indexes) without significantly altering the column definition
    #    or requiring complex migration scripts down the line.
    #
    # 3. It serves as clear, self-documenting code that performance optimization
    #    has been deliberately applied to this specific lookup path, which is
    #    crucial for a high-throughput authentication system.
    #
    # 4. In PostgreSQL and other advanced databases, explicit index names 
    #    prevent the engine from auto-generating opaque index names that can 
    #    complicate performance tuning and query analysis.
    # -------------------------------------------------------------------------
    
    __table_args__ = (
        Index("ix_users_email", "email"),
    )

    # -------------------------------------------------------------------------
    # Primary Key Definition
    # -------------------------------------------------------------------------
    # The surrogate primary key for the User model.
    # We use an auto-incrementing integer for simplicity and performance,
    # as integer joins are generally faster than UUID or string joins.
    # -------------------------------------------------------------------------
    
    id = Column(
        Integer, 
        primary_key=True
    )

    # -------------------------------------------------------------------------
    # Core User Identity Fields
    # -------------------------------------------------------------------------
    # Note: We remove `index=True` from this definition because we have
    # explicitly defined the `Index` construct in `__table_args__` above.
    #
    # The `unique=True` constraint will still typically generate a unique
    # constraint (and often a corresponding index) at the database level, 
    # but our explicit Index ensures that our specific performance 
    # requirements are met and documented.
    # -------------------------------------------------------------------------
    
    email = Column(
        String(255), 
        unique=True, 
        nullable=False
    )

    # -------------------------------------------------------------------------
    # Audit & Tracking Timestamps
    # -------------------------------------------------------------------------
    # These fields are critical for security auditing, analytics, and 
    # determining inactive accounts for potential data retention policies
    # or targeted re-engagement campaigns.
    # -------------------------------------------------------------------------
    
    created_at = Column(
        DateTime(timezone=True), 
        default=lambda: dt.datetime.now(dt.timezone.utc), 
        nullable=False
    )
    
    last_login = Column(
        DateTime(timezone=True), 
        nullable=True
    )

    # -------------------------------------------------------------------------
    # Account Status Flags
    # -------------------------------------------------------------------------
    # Used to manage account lifecycle and access control.
    # Unverified accounts may be periodically purged if they do not
    # complete the onboarding flow within a designated timeframe.
    # -------------------------------------------------------------------------
    
    is_verified = Column(
        Boolean, 
        default=True, 
        nullable=False
    )

    # -------------------------------------------------------------------------
    # ORM Relationships
    # -------------------------------------------------------------------------
    # We define bidirectional relationships with other core entities here.
    # The `cascade="all, delete-orphan"` parameter is crucial for maintaining
    # referential integrity and preventing orphaned records in the database
    # if a user account is deleted (e.g., for GDPR compliance).
    # -------------------------------------------------------------------------
    
    cases = relationship(
        "Case", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    
    preferences = relationship(
        "UserPreference", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )

    # -------------------------------------------------------------------------
    # Helper Methods & Properties
    # -------------------------------------------------------------------------
    # These utility functions encapsulate common operations related to the
    # User model, keeping business logic clean and localized.
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        String representation of the User model.
        Useful for debugging, logging, and interactive console sessions.
        
        Returns:
            str: A formatted string containing the user's ID, email, and status.
        """
        return f"<User(id={self.id}, email='{self.email}', verified={self.is_verified})>"

    def to_dict(self) -> dict:
        """
        Convert the User model to a dictionary representation.
        Useful for API responses, JSON serialization, and caching.
        
        Note: This method deliberately excludes sensitive information
        and related collections to prevent accidental data leaks or
        N+1 query problems.
        
        Returns:
            dict: A dictionary containing the user's core attributes.
        """
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_verified": self.is_verified
        }

    @property
    def has_logged_in(self) -> bool:
        """
        Check if the user has ever successfully logged into the system.
        This can be used to determine if the user is a completely new
        registration or a returning user.
        
        Returns:
            bool: True if last_login is set, False otherwise.
        """
        return self.last_login is not None

    def update_last_login(self) -> None:
        """
        Update the last_login timestamp to the current UTC time.
        
        This should be called during the authentication flow upon
        successful verification of credentials or OTP.
        
        Note: This method modifies the object in memory but does NOT 
        commit the transaction to the database. The caller is responsible 
        for committing the active SQLAlchemy session.
        """
        self.last_login = dt.datetime.now(dt.timezone.utc)

    def verify(self) -> None:
        """
        Mark the user account as verified.
        
        This is typically called after the user successfully enters
        an OTP sent to their email address.
        
        Note: This method modifies the object in memory but does NOT 
        commit the transaction to the database. The caller is responsible 
        for committing the active SQLAlchemy session.
        """
        self.is_verified = True

    # -------------------------------------------------------------------------
    # End of User Model Definition
    # -------------------------------------------------------------------------


class OTPVerification(Base):
    """Model for storing email OTP codes for authentication"""
    __tablename__ = "otp_verifications"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False, index=True)
    otp_hash = Column(String(255), nullable=False)  # Hashed OTP code
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    failed_attempts = Column(Integer, default=0, nullable=False)  # Track failed verification attempts
    locked_until = Column(DateTime(timezone=True), nullable=True)  # Timestamp until which OTP is locked

    def __repr__(self):
        return f"<OTPVerification(email={self.email}, expires_at={self.expires_at})>"
    
    def is_locked(self) -> bool:
        """Check if OTP verification is temporarily locked due to too many failed attempts"""
        if self.locked_until is None:
            return False
        
        now = dt.datetime.now(dt.timezone.utc)
        locked_until = self.locked_until
        
        # Handle naive datetime (from database)
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=dt.timezone.utc)
        
        return now < locked_until


class RevokedToken(Base):
    """Model for storing revoked JWT tokens (logout blacklist)"""
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True)
    jti = Column(String(255), unique=True, nullable=False, index=True)  # JWT ID
    revoked_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)  # When the token would naturally expire

    def __repr__(self):
        return f"<RevokedToken(jti={self.jti})>"


class CaseStatus(str, enum.Enum):
    """Status of a case"""
    ACTIVE = "active"
    APPEALED = "appealed"
    CLOSED = "closed"
    PENDING = "pending"


class DocumentType(str, enum.Enum):
    """Type of legal document"""
    FIR = "FIR"
    CHARGESHEET = "ChargeSheet"
    JUDGMENT = "Judgment"
    APPEAL = "Appeal"
    ORDER = "Order"
    OTHER = "Other"


class Case(Base):
    """Model for tracking user cases"""
    __tablename__ = "cases"
    __table_args__ = (UniqueConstraint("user_id", "case_number", name="uq_user_case_number"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    case_number = Column(String(255), nullable=False)  # User-facing identifier
    case_type = Column(String(255), nullable=False, index=True)  # civil, criminal, family, etc.
    jurisdiction = Column(String(255), nullable=False, index=True)
    status = Column(SQLEnum(CaseStatus), default=CaseStatus.ACTIVE, nullable=False)
    title = Column(String(255), nullable=True)  # Optional case title
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationships
    user = relationship("User", back_populates="cases")
    documents = relationship("CaseDocument", back_populates="case", cascade="all, delete-orphan", order_by="CaseDocument.uploaded_at")
    deadlines = relationship("CaseDeadline", back_populates="case", cascade="all, delete-orphan")
    timeline_events = relationship("CaseTimeline", back_populates="case", cascade="all, delete-orphan")
    attachments = relationship("Attachment", back_populates="case", cascade="all, delete-orphan", order_by="Attachment.uploaded_at")

    def __repr__(self):
        return f"<Case(case_number={self.case_number}, status={self.status})>"


class CaseDocument(Base):
    """Model for storing documents uploaded for a case"""
    __tablename__ = "case_documents"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    document_type = Column(SQLEnum(DocumentType), nullable=False)
    document_content = Column(Text, nullable=True)  # Extracted text from PDF
    file_path = Column(String(255), nullable=True)  # Optional: path to stored PDF
    uploaded_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    summary = Column(Text, nullable=True)  # LLM-generated 3-bullet summary
    remedies = Column(JSON, nullable=True)  # JSON: appeal info, deadlines, costs

    # Relationships
    case = relationship("Case", back_populates="documents")

    def __repr__(self):
        return f"<CaseDocument(case_id={self.case_id}, type={self.document_type})>"


class Attachment(Base):
    """Model for storing uploaded attachments/evidence linked to cases or deadlines"""
    __tablename__ = "attachments"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=True, index=True)
    deadline_id = Column(Integer, ForeignKey("case_deadlines.id", ondelete="CASCADE"), nullable=True, index=True)
    original_filename = Column(String(255), nullable=False)
    stored_path = Column(String(1024), nullable=False)  # Absolute path on disk
    content_type = Column(String(255), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    # Relationships
    case = relationship("Case", back_populates="attachments")
    deadline = relationship("CaseDeadline", back_populates="attachments")

    def __repr__(self):
        return f"<Attachment(id={self.id}, user_id={self.user_id}, filename={self.original_filename})>"


class CaseTimeline(Base):
    """Model for tracking timeline events in a case"""
    __tablename__ = "case_timeline"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(255), nullable=False, index=True)  # document_upload, deadline_created, action_completed, etc.
    event_date = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True)
    description = Column(Text, nullable=False)
    event_metadata = Column(JSON, nullable=True)  # Extra context (document_id, deadline_id, etc.)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    # Relationships
    case = relationship("Case", back_populates="timeline_events")

    def __repr__(self):
        return f"<CaseTimeline(case_id={self.case_id}, event_type={self.event_type})>"


class CasePresence(Base):
    """Tracks which users are currently viewing or editing a case"""
    __tablename__ = "case_presence"
    __table_args__ = (
        UniqueConstraint("case_id", "user_id", name="uq_case_user_presence"),
        {"extend_existing": True},
    )

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(50), default="online", nullable=False)  # online, away, offline
    last_seen_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    case = relationship("Case", backref="presence_entries")
    user = relationship("User", backref="presence_entries")

    def __repr__(self):
        return f"<CasePresence(case_id={self.case_id}, user_id={self.user_id}, status={self.status})>"


# Database initialization
def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


@contextmanager
def db_session():
    """
    Context manager for database sessions.
    Ensures the session is closed after use, even if an exception occurs.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    """
    Generator that yields a database session and ensures it's closed after use.
    Suitable for use as a FastAPI dependency or context manager.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== Helper Functions ====================


def create_or_update_user_preference(
    db: Session,
    user_id: int,
    email: str,
    phone_number: Optional[str] = None,
    notification_channel: NotificationChannel = NotificationChannel.BOTH,
    timezone: str = "UTC",
    # Holiday-aware reminder engine (MVP)
    holiday_aware_reminders: bool = False,
    holiday_country: Optional[str] = None,
    holiday_region: Optional[str] = None,
    holiday_calendar_json: Optional[str] = None,
) -> UserPreference:
    """Create or update user notification preferences"""
    pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()

    if pref:
        pref.email = email
        pref.phone_number = phone_number
        pref.notification_channel = notification_channel
        pref.timezone = timezone
        # Holiday-aware reminder engine (MVP)
        pref.holiday_aware_reminders = holiday_aware_reminders
        pref.holiday_country = holiday_country
        pref.holiday_region = holiday_region
        pref.holiday_calendar_json = holiday_calendar_json
        pref.updated_at = dt.datetime.now(dt.timezone.utc)

    else:
        pref = UserPreference(
            user_id=user_id,
            email=email,
            phone_number=phone_number,
            notification_channel=notification_channel,
            timezone=timezone,
            # Holiday-aware reminder engine (MVP)
            holiday_aware_reminders=holiday_aware_reminders,
            holiday_country=holiday_country,
            holiday_region=holiday_region,
            holiday_calendar_json=holiday_calendar_json,
        )

        db.add(pref)
    
    db.commit()
    db.refresh(pref)
    return pref


def create_case_deadline(
    db: Session,
    user_id: int,
    case_id: int,
    case_title: str,
    deadline_date: dt.datetime,
    deadline_type: str,
    description: Optional[str] = None,
) -> CaseDeadline:
    """Create a new case deadline.

    Security: enforce that `case_id` belongs to `user_id` (server-side ownership validation).
    """
    try:
        normalized_case_id = int(case_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("case_id must be an integer matching cases.id") from exc

    # Ownership validation (prevents creating deadlines for other users' cases)
    case = db.query(Case).filter(Case.id == normalized_case_id).first()
    if not case or case.user_id != user_id:
        raise PermissionError(
            "case_id not found or not owned by the provided user_id"
        )

    deadline = CaseDeadline(
        user_id=user_id,
        case_id=normalized_case_id,
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
    now = dt.datetime.now(dt.timezone.utc)
    target_date = dt.datetime.fromtimestamp(now.timestamp() + (days_before * 86400), tz=dt.timezone.utc)
    
    return db.query(CaseDeadline).filter(
        CaseDeadline.is_completed == False,
        CaseDeadline.deadline_date <= target_date,
        CaseDeadline.deadline_date > now,
    ).all()


def get_user_deadlines(db: Session, user_id: int) -> List[CaseDeadline]:
    """Get all active deadlines for a user"""
    now = dt.datetime.now(dt.timezone.utc)
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
    user_id: int,
    channel: NotificationChannel,
    recipient: str,
    days_before: int,
    status: NotificationStatus = NotificationStatus.PENDING,
    message_id: Optional[str] = None,
    error_message: Optional[str] = None,
    message_preview: Optional[str] = None,
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
        message_preview=message_preview,
        sent_at=dt.datetime.now(dt.timezone.utc) if status != NotificationStatus.PENDING else None,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_notification_history(db: Session, user_id: int, limit: int = 50) -> List[NotificationLog]:
    """Get notification history for a user"""
    return db.query(NotificationLog).filter(
        NotificationLog.user_id == user_id
    ).order_by(NotificationLog.created_at.desc()).limit(limit).all()


# ==================== Analytics & Case Tracking Helper Functions ====================


def create_case_record(
    db: Session,
    hashed_case_id: str,
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
        hashed_case_id=hashed_case_id,
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
    hashed_case_id: str,
    appeal_filed: bool = False,
    appeal_date: Optional[dt.datetime] = None,
    appeal_outcome: Optional[str] = None,
    appeal_success: Optional[bool] = None,
    time_to_appeal_verdict: Optional[int] = None,
    appeal_cost: Optional[str] = None,
) -> CaseOutcome:
    """Update case outcome with appeal information"""
    case = db.query(CaseRecord).filter(CaseRecord.hashed_case_id == hashed_case_id).first()
    if not case:
        raise ValueError(f"Case {hashed_case_id} not found")
    
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


def get_case_record(db: Session, hashed_case_id: str) -> Optional[CaseRecord]:
    """Get a case record by ID"""
    return db.query(CaseRecord).filter(CaseRecord.hashed_case_id == hashed_case_id).first()


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
    user_id: int,
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


def get_user_feedback(db: Session, user_id: int, limit: int = 50) -> List[UserFeedback]:
    """Get feedback submitted by a user"""
    return db.query(UserFeedback).filter(
        UserFeedback.user_id == user_id
    ).order_by(UserFeedback.created_at.desc()).limit(limit).all()


def submit_model_feedback(
    db: Session,
    user_id: str,
    model_name: str,
    task: str,
    case_id: Optional[int] = None,
    is_accurate: Optional[bool] = None,
    corrected_text: Optional[str] = None,
    feedback_notes: Optional[str] = None,
) -> ModelFeedback:
    """Persist model output feedback for training and evaluation"""
    fb = ModelFeedback(
        user_id=str(user_id),
        model_name=model_name,
        task=task,
        case_id=case_id,
        is_accurate=is_accurate,
        corrected_text=corrected_text,
        feedback_notes=feedback_notes,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return fb


def aggregate_model_performance(db: Session, task: Optional[str] = None) -> List[ModelPerformance]:
    """Compute simple model performance aggregates from `model_feedback` rows.

    This is a lightweight aggregator used by the dashboard and can be run periodically.
    Returns newly-created/updated ModelPerformance rows (in-memory list) but does NOT persist them automatically.
    """
    query = db.query(ModelFeedback)
    if task:
        query = query.filter(ModelFeedback.task == task)

    rows = query.all()
    stats = {}
    for r in rows:
        key = (r.model_name, r.task, getattr(r.case, "case_type", None), getattr(r.case, "jurisdiction", None))
        if key not in stats:
            stats[key] = {"samples": 0, "accurate": 0}
        stats[key]["samples"] += 1
        if r.is_accurate:
            stats[key]["accurate"] += 1

    results = []
    for (model_name, task_name, case_type, jurisdiction), v in stats.items():
        samples = v["samples"]
        accurate = v["accurate"]
        accuracy = f"{(accurate / samples * 100):.1f}%" if samples > 0 else "0%"
        mp = ModelPerformance(
            model_name=model_name,
            task=task_name,
            case_type=case_type,
            jurisdiction=jurisdiction,
            samples=samples,
            accurate_count=accurate,
            accuracy=accuracy,
        )
        results.append(mp)

    return results


def submit_similarity_feedback(
    db: Session,
    user_id: str,
    candidate_case_id: int,
    query_signature: str,
    relevance: bool,
) -> SimilarityFeedback:
    """Persist feedback for a similarity search result"""
    feedback = SimilarityFeedback(
        user_id=str(user_id),
        candidate_case_id=candidate_case_id,
        query_signature=query_signature,
        relevance=relevance,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def get_similarity_feedback(
    db: Session,
    user_id: Optional[str] = None,
    query_signature: Optional[str] = None,
    candidate_case_id: Optional[int] = None,
    limit: int = 100,
) -> List[SimilarityFeedback]:
    """Get similarity feedback rows filtered by user, query, or candidate case"""
    query = db.query(SimilarityFeedback)

    if user_id is not None:
        query = query.filter(SimilarityFeedback.user_id == str(user_id))
    if query_signature is not None:
        query = query.filter(SimilarityFeedback.query_signature == query_signature)
    if candidate_case_id is not None:
        query = query.filter(SimilarityFeedback.candidate_case_id == candidate_case_id)

    return query.order_by(SimilarityFeedback.created_at.desc()).limit(limit).all()


# ==================== User & Authentication Helper Functions ====================


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email address"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str) -> User:
    """Create a new user"""
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_last_login(db: Session, user_id: int) -> User:
    """Update user's last login timestamp"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(user)
    return user


def create_otp_verification(
    db: Session,
    email: str,
    otp_hash: str,
    expires_at: dt.datetime,
    max_requests_per_hour: int = 5,
) -> OTPVerification:
    """Create a new OTP verification record with rate limiting"""
    # Check recent OTPs
    one_hour_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    recent_otps = db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.created_at > one_hour_ago,
    ).count()

    if recent_otps >= max_requests_per_hour:
        raise ValueError("Too many OTP requests. Please try again later.")

    otp = OTPVerification(
        email=email,
        otp_hash=otp_hash,
        expires_at=expires_at,
    )
    db.add(otp)
    db.commit()
    db.refresh(otp)
    return otp


def get_pending_otp(db: Session, email: str) -> Optional[OTPVerification]:
    """Get unused, non-expired OTP for email"""
    now = dt.datetime.now(dt.timezone.utc)
    return db.query(OTPVerification).filter(
        OTPVerification.email == email,
        OTPVerification.is_used == False,
        OTPVerification.expires_at > now,
    ).order_by(OTPVerification.created_at.desc()).first()


def mark_otp_as_used(db: Session, otp_id: int) -> bool:
    """Mark an OTP as used"""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.is_used = True
        db.commit()
        return True
    return False


def record_otp_failed_attempt(db: Session, otp_id: int, lockout_duration_minutes: int = 15, max_failed_attempts: int = 5) -> bool:
    """
    Record a failed OTP verification attempt and implement lockout after max attempts.
    
    Args:
        db: Database session
        otp_id: OTP record ID
        lockout_duration_minutes: Minutes to lock OTP after max attempts exceeded
        max_failed_attempts: Maximum failed attempts before lockout
    
    Returns:
        True if updated, False if OTP not found
    """
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.failed_attempts += 1
        
        # Lock OTP if max attempts exceeded
        if otp.failed_attempts >= max_failed_attempts:
            otp.locked_until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=lockout_duration_minutes)
            logger.warning(
                f"OTP for {otp.email} locked after {otp.failed_attempts} failed attempts. "
                f"Locked until {otp.locked_until}"
            )
        
        db.commit()
        db.refresh(otp)
        return True
    return False


def reset_otp_failed_attempts(db: Session, otp_id: int) -> bool:
    """Reset failed attempt counter on successful verification"""
    otp = db.query(OTPVerification).filter(OTPVerification.id == otp_id).first()
    if otp:
        otp.failed_attempts = 0
        otp.locked_until = None
        db.commit()
        db.refresh(otp)
        return True
    return False


def cleanup_expired_otps(db: Session) -> int:
    """Delete expired OTPs, return count of deleted"""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(OTPVerification).filter(
        OTPVerification.expires_at < now
    ).delete()
    db.commit()
    return deleted


def revoke_token(db: Session, jti: str, expires_at: dt.datetime) -> RevokedToken:
    """Add a token JTI to the revoked list"""
    token = RevokedToken(jti=jti, expires_at=expires_at)
    db.add(token)
    db.commit()
    db.refresh(token)
    return token


def is_token_revoked(db: Session, jti: str) -> bool:
    """Check if a token has been revoked"""
    return db.query(RevokedToken).filter(RevokedToken.jti == jti).first() is not None


def cleanup_expired_revoked_tokens(db: Session) -> int:
    """Delete revoked tokens that have naturally expired"""
    now = dt.datetime.now(dt.timezone.utc)
    deleted = db.query(RevokedToken).filter(
        RevokedToken.expires_at < now
    ).delete()
    db.commit()
    return deleted


# ==================== Case Management Helper Functions ====================


def create_case(
    db: Session,
    user_id: int,
    case_number: str,
    case_type: str,
    jurisdiction: str,
    title: Optional[str] = None,
    status: CaseStatus = CaseStatus.ACTIVE,
) -> Case:
    """Create a new case"""
    case = Case(
        user_id=user_id,
        case_number=case_number,
        case_type=case_type,
        jurisdiction=jurisdiction,
        title=title,
        status=status,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def get_user_cases(db: Session, user_id: int, include_closed: bool = True) -> List[Case]:
    """Get all cases for a user"""
    query = db.query(Case).filter(Case.user_id == user_id)
    if not include_closed:
        query = query.filter(Case.status != CaseStatus.CLOSED)
    return query.order_by(Case.created_at.desc()).all()


def get_case_by_id(db: Session, case_id: int) -> Optional[Case]:
    """Get a case by ID"""
    return db.query(Case).filter(Case.id == case_id).first()


def get_case_by_number(db: Session, user_id: int, case_number: str) -> Optional[Case]:
    """Get a case by case number for a specific user"""
    return db.query(Case).filter(
        Case.user_id == user_id,
        Case.case_number == case_number,
    ).first()

def get_notification_template_for_user(db: Session, user_id: int) -> Optional["NotificationTemplate"]:
    return db.query(NotificationTemplate).filter(NotificationTemplate.user_id == user_id).first()

def create_or_update_notification_template(
    db: Session,
    user_id: int,
    sms_template: Optional[str] = None,
    email_subject_template: Optional[str] = None,
    email_html_template: Optional[str] = None,
):
    tmpl = db.query(NotificationTemplate).filter(NotificationTemplate.user_id == user_id).first()
    if tmpl:
        if sms_template is not None:
            tmpl.sms_template = sms_template
        if email_subject_template is not None:
            tmpl.email_subject_template = email_subject_template
        if email_html_template is not None:
            tmpl.email_html_template = email_html_template
        tmpl.updated_at = dt.datetime.now(dt.timezone.utc)
    else:
        tmpl = NotificationTemplate(
            user_id=user_id,
            sms_template=sms_template,
            email_subject_template=email_subject_template,
            email_html_template=email_html_template,
        )
        db.add(tmpl)

    db.commit()
    db.refresh(tmpl)
    return tmpl


def update_case_status(db: Session, case_id: int, status: CaseStatus) -> Optional[Case]:
    """Update case status"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if case:
        case.status = status
        case.updated_at = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(case)
    return case


def delete_case(db: Session, case_id: int) -> bool:
    """Delete a case and all related data"""
    case = db.query(Case).filter(Case.id == case_id).first()
    if case:
        db.delete(case)
        db.commit()
        return True
    return False


# ==================== Case Document Helper Functions ====================


def create_case_document(
    db: Session,
    case_id: int,
    document_type: DocumentType,
    user_id: int,
    document_content: Optional[str] = None,
    file_path: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
) -> CaseDocument:
    """Create a new case document.

    Security: enforce that `case_id` belongs to `user_id` (server-side ownership
    validation), consistent with create_case_deadline.
    """
    try:
        normalized_case_id = int(case_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("case_id must be an integer matching cases.id") from exc

    # Ownership validation (prevents attaching documents to another user's case)
    case = db.query(Case).filter(Case.id == normalized_case_id).first()
    if not case or case.user_id != user_id:
        raise PermissionError(
            "case_id not found or not owned by the provided user_id"
        )

    doc = CaseDocument(
        case_id=normalized_case_id,
        document_type=document_type,
        document_content=document_content,
        file_path=file_path,
        summary=summary,
        remedies=remedies,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_case_documents(db: Session, case_id: int) -> List[CaseDocument]:
    """Get all documents for a case"""
    return db.query(CaseDocument).filter(
        CaseDocument.case_id == case_id
    ).order_by(CaseDocument.uploaded_at).all()


def get_case_document_by_id(db: Session, document_id: int) -> Optional[CaseDocument]:
    """Get a case document by ID"""
    return db.query(CaseDocument).filter(CaseDocument.id == document_id).first()


def update_case_document(
    db: Session,
    document_id: int,
    document_content: Optional[str] = None,
    summary: Optional[str] = None,
    remedies: Optional[dict] = None,
) -> Optional[CaseDocument]:
    """Update case document"""
    doc = db.query(CaseDocument).filter(CaseDocument.id == document_id).first()
    if doc:
        if document_content is not None:
            doc.document_content = document_content
        if summary is not None:
            doc.summary = summary
        if remedies is not None:
            doc.remedies = remedies
        db.commit()
        db.refresh(doc)
    return doc


def create_attachment(
    db: Session,
    user_id: int,
    original_filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    case_id: Optional[int] = None,
    deadline_id: Optional[int] = None,
) -> "Attachment":
    """Create a new attachment record linked to a case or deadline"""
    att = Attachment(
        user_id=user_id,
        case_id=case_id,
        deadline_id=deadline_id,
        original_filename=original_filename,
        stored_path=stored_path,
        content_type=content_type,
        size_bytes=size_bytes,
    )
    db.add(att)
    db.commit()
    db.refresh(att)
    return att


def get_attachments_for_case(db: Session, case_id: int):
    return db.query(Attachment).filter(Attachment.case_id == case_id).order_by(Attachment.uploaded_at.desc()).all()


def get_attachments_for_deadline(db: Session, deadline_id: int):
    return db.query(Attachment).filter(Attachment.deadline_id == deadline_id).order_by(Attachment.uploaded_at.desc()).all()


def get_attachment_by_id(db: Session, attachment_id: int):
    return db.query(Attachment).filter(Attachment.id == attachment_id).first()


def delete_attachment(db: Session, attachment_id: int) -> bool:
    att = db.query(Attachment).filter(Attachment.id == attachment_id).first()
    if not att:
        return False
    db.delete(att)
    db.commit()
    return True


# ==================== Case Timeline Helper Functions ====================


def create_timeline_event(
    db: Session,
    case_id: int,
    event_type: str,
    description: str,
    event_date: Optional[dt.datetime] = None,
    metadata: Optional[dict] = None,
) -> CaseTimeline:
    """Create a new timeline event"""
    event = CaseTimeline(
        case_id=case_id,
        event_type=event_type,
        description=description,
        event_date=event_date or dt.datetime.now(dt.timezone.utc),
        event_metadata=metadata,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def get_case_timeline(db: Session, case_id: int) -> List[CaseTimeline]:
    """Get timeline events for a case"""
    return db.query(CaseTimeline).filter(
        CaseTimeline.case_id == case_id
    ).order_by(CaseTimeline.event_date.desc()).all()


def get_user_stats(db: Session, user_id: int) -> dict:
    """Get statistics for a user's cases"""
    cases = get_user_cases(db, user_id)

    active_count = len([c for c in cases if c.status == CaseStatus.ACTIVE])
    appealed_count = len([c for c in cases if c.status == CaseStatus.APPEALED])
    closed_count = len([c for c in cases if c.status == CaseStatus.CLOSED])

    # Get upcoming deadlines
    now = dt.datetime.now(dt.timezone.utc)
    upcoming_deadlines = db.query(CaseDeadline).filter(
        CaseDeadline.user_id == user_id,
        CaseDeadline.is_completed == False,
        CaseDeadline.deadline_date > now,
    ).count()

    return {
        "total_cases": len(cases),
        "active_cases": active_count,
        "appealed_cases": appealed_count,
        "closed_cases": closed_count,
        "upcoming_deadlines": upcoming_deadlines,
    }
