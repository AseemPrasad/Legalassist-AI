import datetime as dt
import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
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
    case_id = Column(String(255), nullable=False)
    report_type = Column(String(50), nullable=True)
    format = Column(String(50), nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Report(report_id={self.report_id}, status={self.status})>"
