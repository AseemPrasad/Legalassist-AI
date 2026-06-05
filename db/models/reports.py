import datetime as dt
import enum
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from db.base import Base


class ReportStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportType(str, enum.Enum):
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    LEGAL_BRIEF = "legal_brief"


class ReportFormat(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True)
    report_id = Column(String(255), unique=True, index=True, nullable=False)
    job_id = Column(String(255), index=True, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    report_type = Column(String(50), nullable=True)
    format = Column(String(50), nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    file_path = Column(String(1024), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Report(report_id={self.report_id}, status={self.status})>"
