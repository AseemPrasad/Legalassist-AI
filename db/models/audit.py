import datetime as dt

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship

from db.base import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_case_id_occurred_at", "case_id", "occurred_at"),
        Index("ix_audit_events_actor_id_occurred_at", "actor_user_id", "occurred_at"),
    )

    id = Column(Integer, primary_key=True)
    actor = Column(String(255), nullable=False)
    actor_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(255), nullable=False, index=True)
    resource = Column(String(255), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=True, index=True)
    occurred_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True)
    event_metadata = Column("metadata", JSON, nullable=True)
    notes = Column(Text, nullable=True)

    actor_user = relationship("db.models.auth.User")
    case = relationship("Case")
