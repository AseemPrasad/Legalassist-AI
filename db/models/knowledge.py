import datetime as dt
import enum

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Index
from sqlalchemy.orm import relationship

from db.base import Base


class KnowledgeInvalidationStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class KnowledgeInvalidation(Base):
    __tablename__ = "knowledge_invalidations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=True, index=True)
    document_id = Column(Integer, ForeignKey("case_documents.id", ondelete="CASCADE"), nullable=True, index=True)
    scope_type = Column(String(50), nullable=False, index=True)
    scope_value = Column(String(255), nullable=False, index=True)
    reason = Column(String(255), nullable=False, index=True)
    details = Column(JSON, nullable=True)
    status = Column(String(50), default=KnowledgeInvalidationStatus.PENDING.value, nullable=False, index=True)
    invalidated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=True, index=True)
    recompute_started_at = Column(DateTime(timezone=True), nullable=True)
    recompute_completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    recompute_attempts = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    user = relationship("User")
    case = relationship("Case")
    document = relationship("CaseDocument")

    __table_args__ = (
        Index("ix_knowledge_invalidations_scope", "scope_type", "scope_value", "status"),
    )

    def __repr__(self):
        return (
            f"<KnowledgeInvalidation(scope={self.scope_type}:{self.scope_value}, "
            f"reason={self.reason}, status={self.status})>"
        )
