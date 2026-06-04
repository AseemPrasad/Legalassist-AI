"""Database model for distributed lock audit records."""

import datetime as dt
import enum
from sqlalchemy import Column, Integer, String, DateTime, Text, Enum as SQLEnum
from db.base import Base


class LockAction(str, enum.Enum):
    ACQUIRED = "acquired"
    RELEASED = "released"
    EXTENDED = "extended"
    FAILED = "failed"
    EXPIRED = "expired"


class DocumentProcessingLock(Base):
    """Audit log of distributed lock events for a document_id."""

    __tablename__ = "document_processing_locks"

    id = Column(Integer, primary_key=True)
    document_id = Column(String(255), nullable=False, index=True)
    task_id = Column(String(255), nullable=True, index=True)
    worker_id = Column(String(255), nullable=True)
    action = Column(SQLEnum(LockAction), nullable=False)
    lock_key = Column(String(255), nullable=False)
    ttl_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    def __repr__(self):
        return f"<DocumentProcessingLock(document_id={self.document_id}, action={self.action})>"