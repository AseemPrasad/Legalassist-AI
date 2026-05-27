import datetime as dt
import enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Enum as SQLEnum,
    UniqueConstraint,
    Index,
    JSON,
)
from sqlalchemy.orm import relationship
from db.base import Base


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


class CaseDeadline(Base):
    __tablename__ = "case_deadlines"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    case_title = Column(String(255), nullable=False)
    deadline_date = Column(DateTime(timezone=True), nullable=False, index=True)
    deadline_type = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))
    is_completed = Column(Boolean, default=False, index=True)

    case = relationship("Case", back_populates="deadlines")
    notifications = relationship("NotificationLog", back_populates="deadline")
    attachments = relationship("Attachment", back_populates="deadline")

    def days_until_deadline(self) -> int:
        from db.session import _to_utc_datetime
        now = dt.datetime.now(dt.timezone.utc)
        deadline = _to_utc_datetime(self.deadline_date)
        
        # Calculate calendar days difference in UTC to avoid 1-day errors 
        # caused by truncation of fractional days in timedelta.days.
        delta_days = (deadline.date() - now.date()).days
        return max(0, delta_days)


class Case(Base):
    __tablename__ = "cases"
    __table_args__ = (
        UniqueConstraint("user_id", "case_number", name="uq_user_case_number"),
        Index("ix_cases_anonymized_id", "anonymized_id"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    case_number = Column(String(255), nullable=False)
    case_type = Column(String(255), nullable=False, index=True)
    jurisdiction = Column(String(255), nullable=False, index=True)
    status = Column(SQLEnum(CaseStatus), default=CaseStatus.ACTIVE, nullable=False)
    title = Column(String(255), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    # Stores the HMAC-derived anonymized share ID so lookups by anon_id are O(log N).
    # Populated on first call to generate_anonymized_case_data().
    anonymized_id = Column(String(64), nullable=True, unique=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    __mapper_args__ = {
        "version_id_col": version
    }

    user = relationship("User", back_populates="cases")
    documents = relationship("CaseDocument", back_populates="case", cascade="all, delete-orphan")
    deadlines = relationship("CaseDeadline", back_populates="case", cascade="all, delete-orphan")
    timeline_events = relationship("CaseTimeline", back_populates="case", cascade="all, delete-orphan")
    notes = relationship("CaseNote", back_populates="case", cascade="all, delete-orphan")
    attachments = relationship("Attachment", back_populates="case", cascade="all, delete-orphan")


class CaseDocument(Base):
    __tablename__ = "case_documents"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    source_attachment_id = Column(Integer, ForeignKey("attachments.id", ondelete="SET NULL"), nullable=True, index=True)
    document_type = Column(SQLEnum(DocumentType), nullable=False)
    document_content = Column(Text, nullable=True)
    file_path = Column(String(255), nullable=True)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    summary = Column(Text, nullable=True)
    remedies = Column(JSON, nullable=True)
    extracted_metadata = Column(JSON, nullable=True)
    extraction_method = Column(String(50), nullable=True)
    ocr_used = Column(Boolean, default=False, nullable=False)

    case = relationship("Case", back_populates="documents")
    attachment = relationship("Attachment", foreign_keys=[source_attachment_id])


class Attachment(Base):
    __tablename__ = "attachments"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=True, index=True)
    deadline_id = Column(Integer, ForeignKey("case_deadlines.id", ondelete="CASCADE"), nullable=True, index=True)
    document_id = Column(Integer, ForeignKey("case_documents.id", ondelete="SET NULL"), nullable=True, index=True)
    original_filename = Column(String(255), nullable=False)
    stored_path = Column(String(1024), nullable=False)
    content_type = Column(String(255), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    is_encrypted = Column(Boolean, default=False, nullable=False)
    wrapped_key = Column(Text, nullable=True)

    case = relationship("Case", back_populates="attachments")
    deadline = relationship("CaseDeadline", back_populates="attachments")


class CaseTimeline(Base):
    __tablename__ = "case_timeline"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(255), nullable=False, index=True)
    event_date = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True)
    description = Column(Text, nullable=False)
    event_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    case = relationship("Case", back_populates="timeline_events")


class CaseNote(Base):
    __tablename__ = "case_notes"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    draft_text = Column(Text, nullable=False, default="")
    published_text = Column(Text, nullable=True)
    published_version_id = Column(Integer, nullable=True)
    draft_updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    case = relationship("Case", back_populates="notes")
    versions = relationship("CaseNoteVersion", back_populates="note", cascade="all, delete-orphan", order_by="CaseNoteVersion.version_number")


class CaseNoteVersion(Base):
    __tablename__ = "case_note_versions"

    id = Column(Integer, primary_key=True)
    note_id = Column(Integer, ForeignKey("case_notes.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    note_text = Column(Text, nullable=False)
    change_type = Column(String(50), nullable=False, default="published")
    changed_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    changed_by_email = Column(String(255), nullable=True)
    version_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    note = relationship("CaseNote", back_populates="versions")
