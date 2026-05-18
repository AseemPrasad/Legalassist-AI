"""
Database models for Report tracking and lifecycle management.

This module defines the Report model which stores metadata about generated
reports including Celery task IDs, file paths, and status for reliable
tracking and file retrieval.
"""

import datetime as dt
import enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from db.base import Base


class ReportStatus(str, enum.Enum):
    """Status of a report generation"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportType(str, enum.Enum):
    """Type of report"""
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    LEGAL_BRIEF = "legal_brief"


class ReportFormat(str, enum.Enum):
    """Output format for report"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"


class Report(Base):
    """
    Stores metadata for generated reports.
    
    This table enables:
    - Reliable tracking of report status via stored celery_task_id
    - Direct file path retrieval without glob patterns
    - User ownership validation for downloads
    - Case-to-report associations
    """
    __tablename__ = "reports"
    __table_args__ = (
        Index("ix_reports_user_id", "user_id"),
        Index("ix_reports_case_id", "case_id"),
        Index("ix_reports_status", "status"),
        Index("ix_reports_created_at", "created_at"),
        UniqueConstraint("report_id", name="uq_report_id"),
    )

    id = Column(Integer, primary_key=True)
    
    # Report identification
    report_id = Column(String(36), nullable=False, unique=True)  # UUID format
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    
    # Report configuration
    report_type = Column(SQLEnum(ReportType), default=ReportType.COMPREHENSIVE, nullable=False)
    format = Column(SQLEnum(ReportFormat), default=ReportFormat.PDF, nullable=False)
    style = Column(String(50), default="formal", nullable=False)  # formal, casual
    
    # Celery task tracking (the key fix)
    celery_task_id = Column(String(255), nullable=False, unique=True)
    
    # Status tracking
    status = Column(SQLEnum(ReportStatus), default=ReportStatus.PENDING, nullable=False, index=True)
    
    # File path (the key fix - no more globbing needed)
    file_path = Column(Text, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False,
        index=True
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.timezone.utc),
        onupdate=lambda: dt.datetime.now(dt.timezone.utc),
        nullable=False
    )

    # Relationships
    case = relationship("Case", foreign_keys=[case_id])
    user = relationship("User", foreign_keys=[user_id])
